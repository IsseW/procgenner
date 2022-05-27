use std::{
    fs::{self},
    sync::Arc,
};

use bevy::{
    pbr::{PbrBundle, StandardMaterial},
    prelude::{
        error, Assets, Commands, Component, Entity, Handle, Local, Mesh, Query, Res, ResMut, With,
    },
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    utils::HashMap,
};
use bevy_egui::{
    egui::{self, mutex::Mutex, DragValue},
    EguiContext,
};
use building_blocks::{
    core::{Extent3i, PointN},
    mesh::{greedy_quads, GreedyQuadsBuffer, PosNormMesh, RIGHT_HANDED_Y_UP_CONFIG},
    storage::Array3x1,
};
use egui_node_graph::{
    AnyParameterId, GraphEditorState, InputId, InputParamKind, NodeDataTrait, NodeId, NodeResponse,
    NodeTemplateIter as NodeTemplateIterTrait, UserResponseTrait,
};
use futures::Future;
use petgraph::graphmap::DiGraphMap;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use vek::*;

use self::{
    evaluate::{eval, Ctx},
    template::{aabr_ui, DataType, Fill, NodeTemplate, Primitive, ValueType},
};

mod evaluate;
mod template;

#[derive(Serialize, Deserialize)]
pub struct NodeData {
    template: NodeTemplate,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Response {
    Changed,
}

#[derive(Default, Serialize, Deserialize)]
pub struct GraphState {}

struct AllNodeTemplates;

impl NodeTemplateIterTrait for AllNodeTemplates {
    type Item = NodeTemplate;

    fn all_kinds(&self) -> Vec<Self::Item> {
        NodeTemplate::iter().collect()
    }
}

type Graph = egui_node_graph::Graph<NodeData, DataType, ValueType>;

impl UserResponseTrait for Response {}

impl NodeDataTrait for NodeData {
    type Response = Response;

    type UserState = GraphState;

    type DataType = DataType;

    type ValueType = ValueType;

    fn bottom_ui(
        &self,
        _ui: &mut egui::Ui,
        _node_id: NodeId,
        _graph: &Graph,
        _state: &GraphState,
    ) -> Vec<egui_node_graph::NodeResponse<Self::Response>>
    where
        Self::Response: UserResponseTrait,
    {
        Vec::new()
    }
}

type EditorState = GraphEditorState<NodeData, DataType, ValueType, NodeTemplate, GraphState>;

#[derive(Serialize, Deserialize)]
pub struct NodeGraph {
    state: EditorState,
    structure: Structure,
}

impl Default for NodeGraph {
    fn default() -> Self {
        Self {
            state: GraphEditorState::new(1.0, GraphState::default()),
            structure: Structure::default(),
        }
    }
}

fn execute<F: Future<Output = ()> + Send + 'static>(f: F) {
    std::thread::spawn(move || futures::executor::block_on(f));
}

#[derive(Serialize, Deserialize)]
pub struct Structure {
    pub bounds: Aabr<i32>,
    pub alt: i32,
}

impl Default for Structure {
    fn default() -> Self {
        Self {
            bounds: Aabr {
                min: Vec2::broadcast(-5),
                max: Vec2::broadcast(5),
            },
            alt: Default::default(),
        }
    }
}

fn save(graph: &NodeGraph) {
    if let Ok(serialized) = ron::ser::to_string(graph) {
        let task = rfd::AsyncFileDialog::new()
            .add_filter("procgraph", &["procgraph"])
            .set_file_name("structure.procgraph")
            .save_file();
        execute(async move {
            let file = task.await;
            if let Some(file) = file {
                if fs::write(file.path(), serialized.as_bytes()).is_err() {
                    error!("Failed to write to file");
                }
            }
        });
    } else {
        error!("Failed to serialize graph");
    }
}

fn load(new_graph: Arc<Mutex<Option<NodeGraph>>>) {
    let task = rfd::AsyncFileDialog::new()
        .add_filter("procgraph", &["procgraph"])
        .pick_file();
    let new_graph = new_graph.clone();
    execute(async move {
        let file = task.await;
        if let Some(file) = file {
            if let Ok(string) = String::from_utf8(file.read().await) {
                if let Ok(deserialized) = ron::de::from_str::<NodeGraph>(&string) {
                    let mut guard = new_graph.lock();
                    *guard = Some(deserialized);
                } else {
                    error!("Failed to deserialize graph");
                }
            } else {
                error!("Failed to read file");
            }
        }
    });
}

#[derive(Component)]
pub struct GeneratedMeshMarker;

fn create_mesh_bundle(
    mut mesh: PosNormMesh,
    material: Handle<StandardMaterial>,
    meshes: &mut Assets<Mesh>,
) -> PbrBundle {
    assert_eq!(mesh.positions.len(), mesh.normals.len());
    let num_vertices = mesh.positions.len();

    // Bevy might not normalize our surface normals in the vertex shader (before interpolation).
    for n in &mut mesh.normals {
        let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        n[0] = n[0] / norm;
        n[1] = n[1] / norm;
        n[2] = n[2] / norm;
    }

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(mesh.positions),
    );

    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(mesh.normals),
    );
    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        VertexAttributeValues::Float32x2(vec![[0.0; 2]; num_vertices]),
    );
    if mesh.indices.len() > 0 {
        render_mesh.set_indices(Some(Indices::U32(mesh.indices)));
    }

    PbrBundle {
        mesh: meshes.add(render_mesh),
        material,
        ..Default::default()
    }
}

fn display(
    commands: &mut Commands,
    old_meshes: &Query<Entity, With<GeneratedMeshMarker>>,
    graph: &NodeGraph,
    material: &Handle<StandardMaterial>,
    meshes: &mut Assets<Mesh>,
) {
    old_meshes.for_each(|e| commands.entity(e).despawn());
    let mut cache = HashMap::new();
    let ctx = &mut Ctx {
        graph: &graph.state.graph,
        structure: &graph.structure,
        cached: &mut cache,
    };
    let fills: Vec<_> = ctx
        .graph
        .nodes
        .iter()
        .filter_map(|(_, node)| {
            let template = node.user_data.template;
            let input = |i: usize| node.inputs[i].1;
            Some(match template {
                NodeTemplate::FillPrimtive => {
                    let prim = eval::<Primitive>(input(0), ctx)?;
                    let fill = eval::<Fill>(input(1), ctx)?;

                    (prim, fill)
                }
                _ => return None,
            })
        })
        .collect();
    let bounds: Vec<_> = fills.iter().map(|(p, _)| p.bounds()).collect();
    let b = bounds
        .iter()
        .cloned()
        .reduce(|a, b| a.union(b))
        .unwrap_or_default();

    let extent = Extent3i::from_min_and_max(
        PointN::<[i32; 3]>([b.min.x - 1, b.min.y - 1, b.min.z - 1]),
        PointN::<[i32; 3]>([b.max.x + 1, b.max.y + 1, b.max.z + 1]),
    );
    let samples = Array3x1::fill_with(extent, |p| {
        let p = Vec3::new(p.x(), p.y(), p.z());
        fills
            .iter()
            .zip(bounds.iter())
            .find_map(|((prim, f), b)| {
                (b.contains_point(p) && prim.contains_at(p)).then(|| f.sample_at(p))
            })
            .unwrap_or_default()
    });
    let mut buffer = GreedyQuadsBuffer::new(extent, RIGHT_HANDED_Y_UP_CONFIG.quad_groups());
    greedy_quads(&samples, &extent, &mut buffer);
    let mut mesh = PosNormMesh::default();
    for group in buffer.quad_groups.iter() {
        for quad in group.quads.iter() {
            group.face.add_quad_to_pos_norm_mesh(&quad, 1.0, &mut mesh);
        }
    }
    commands
        .spawn_bundle(create_mesh_bundle(mesh, material.clone(), meshes))
        .insert(GeneratedMeshMarker);
}

pub fn ui(
    mut commands: Commands,
    old_meshes: Query<Entity, With<GeneratedMeshMarker>>,
    mut meshes: ResMut<Assets<Mesh>>,
    material: Res<Handle<StandardMaterial>>,
    mut ctx: ResMut<EguiContext>,
    mut graph: Local<NodeGraph>,
    mut map: Local<DiGraphMap<NodeId, InputId>>,
    new_graph: Local<Arc<Mutex<Option<NodeGraph>>>>,
) {
    let mut redraw = false;
    if let Some(new_graph) = new_graph.lock().take() {
        *graph = new_graph;
        redraw = true;
    }
    let response = egui::TopBottomPanel::bottom("Node Graph")
        .min_height(500.0)
        .show(ctx.ctx_mut(), |ui| {
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    save(&graph);
                }
                if ui.button("Load").clicked() {
                    load(new_graph.clone());
                }
                if ui.button("Clear").clicked() {
                    *graph = NodeGraph::default();
                    redraw = true;
                }
            });
            ui.collapsing("Settings", |ui| {
                ui.label("Bounds");
                redraw |= aabr_ui(ui, &mut graph.structure.bounds).changed();
                ui.label("Alt");
                redraw |= ui.add(DragValue::new(&mut graph.structure.alt)).changed();
            });
            graph.state.draw_graph_editor(ui, AllNodeTemplates)
        })
        .inner;
    for response in response.node_responses {
        let update_connection = |graph: &mut Graph, id: InputId| {
            let input = &graph.inputs[id];
            let node_id = input.node;
            let node = &graph.nodes[node_id];
            let t = node.user_data.template.clone();
            t.update_connection(graph, node_id);
        };
        match response {
            NodeResponse::User(response) => {
                match response {
                    Response::Changed => redraw = true,
                }
            }
            NodeResponse::ConnectEventEnded(id) => {
                let graph = &mut graph.state.graph;
                let res = match id {
                    AnyParameterId::Input(id) => {
                        let input = &graph.inputs[id];
                        let node_b = input.node;
                        let output = graph.connections[id];
                        if let Some(node_a) = graph.outputs.get(output).map(|o| o.node) {
                            Some((node_a, node_b, id))
                        } else {
                            None
                        }
                    }
                    AnyParameterId::Output(id) => {
                        let output = &graph.outputs[id];
                        let node_a = output.node;
                        let id = graph
                            .iter_connections()
                            .find_map(|(i, u)| (u == id).then_some(i))
                            .unwrap();
                        if let Some(node_b) = graph.inputs.get(id).map(|i| i.node) {
                            Some((node_a, node_b, id))
                        } else {
                            None
                        }
                    }
                };
                if let Some((node_a, node_b, id)) = res {
                    map.add_edge(node_a, node_b, id);
                    if petgraph::algo::is_cyclic_directed(&*map) {
                        graph.remove_connection(id);
                        map.remove_edge(node_a, node_b);
                    } else {
                        update_connection(graph, id);
                        let i_node = &graph.nodes[node_b];
                        if matches!(
                            i_node.user_data.template,
                            NodeTemplate::Union | NodeTemplate::Intersection
                        ) {
                            graph.add_input_param(
                                node_b,
                                "prim".to_string(),
                                DataType::Primitive,
                                ValueType::Primitive(Primitive::Empty),
                                InputParamKind::ConnectionOnly,
                                true,
                            );
                        }
                        redraw = true;
                    }
                }
            }
            NodeResponse::DisconnectEvent(id) => {
                let g = &mut graph.state.graph;
                if let Some(input) = g.inputs.get(id) {
                    let node_b = input.node;
                    if let Some(node_a) = map
                        .all_edges()
                        .find_map(|(a, _, i)| (*i == id).then_some(a))
                    {
                        map.remove_edge(node_a, node_b);
                    }

                    update_connection(g, id);
                    let i_node = &mut g.nodes[node_b];
                    if matches!(
                        i_node.user_data.template,
                        NodeTemplate::Union | NodeTemplate::Intersection
                    ) {
                        i_node.inputs.retain(|f| f.1 != id);
                        g.inputs.remove(id);
                    }
                }
                redraw = true;
            }
            NodeResponse::CreatedNode(id) => {
                map.add_node(id);
            }
            NodeResponse::DeleteNode(id) => {
                map.remove_node(id);
                redraw = true;
            }
            _ => {}
        }
    }
    if redraw {
        display(&mut commands, &old_meshes, &graph, &material, &mut meshes);
    }
}
