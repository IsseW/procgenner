use super::{Graph, NodeData, Response};
use bevy::utils::HashSet;
use bevy_egui::egui::{self, emath::Numeric, Color32};
use building_blocks::{
    mesh::{IsOpaque, MergeVoxel},
    storage::IsEmpty,
};
use egui_node_graph::{DataTypeTrait, InputParamKind, NodeId, NodeTemplateTrait, WidgetValueTrait};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use strum_macros::{AsRefStr, EnumIter};
use vek::*;

macro_rules! count {
    ($($id:ident),*) => {
        {
            #[allow(dead_code)]
            enum Count {
                $($id,)*
                Last__,
            }
            Count::Last__ as usize
        }
    };
}

macro_rules! types {
    ($($ident:ident($t:ty) $(@$color:ident)?),*;$($trait:ident[$first_impl:ident$(,$impls:ident)*$(,)?]),*$(,)?) => {
        #[derive(AsRefStr, Clone, Copy, Hash, Debug, Serialize, Deserialize)]
        pub enum DataType {
            $($trait),*,
            $($ident),*,
        }
        const NUM_TRAITS: usize = count!($($trait),*);
        const TRAIT_DEFAULT_ARRAY: [DataType; NUM_TRAITS] = [
            $(DataType::$first_impl),*
        ];
        impl DataType {
            pub fn with_default(&self) -> ValueType {
                match self {
                    $(DataType::$ident => ValueType::$ident(Default::default())),*,
                    _ => unreachable!(),
                }
            }

            pub const fn first_impl(&self) -> DataType {
                match self {
                    $(DataType::$trait => DataType::$first_impl),*,
                    s => *s,
                }
            }

            pub const fn is_trait(&self) -> bool {
                matches!(self, $(DataType::$trait)|*)
            }

            pub const fn impls(&self) -> &'static [DataType] {
                match self {
                    $(
                        DataType::$trait => {
                            &[DataType::$first_impl, $(DataType::$impls),*]
                        }
                    )*
                    _ => &[],
                }
            }

            pub const fn derive_order(&self, traity: DataType) -> usize {
                match traity {
                    $(
                        DataType::$trait => {
                            const ARR: &'static [DataType] = DataType::$trait.impls();
                            let mut i = 0;
                            while if i >= ARR.len() { panic!() } else { true } && ARR[i] as u32 != *self as u32 {
                                i += 1;
                            }
                            i
                        }
                    )*
                    _ => 0,
                }
            }
        }

        impl DataTypeTrait for DataType {
            fn data_type_color(&self) -> Color32 {
                match self {
                    $($(DataType::$ident => Color32::$color)?),*,
                    $(DataType::$trait => DataType::$trait.first_impl().data_type_color()),*,
                }
            }

            fn name(&self) -> &str {
                self.as_ref()
            }
        }


        impl PartialEq for DataType {
            fn eq(&self, other: &Self) -> bool {
                *self as u32 == *other as u32 ||
                matches!(
                    (self, other),
                    $(
                        (DataType::$first_impl, DataType::$trait)
                        $(|(DataType::$impls, DataType::$trait))*
                    )|*
                )
            }
        }

        impl Eq for DataType {}

        #[derive(Serialize, Deserialize, Clone)]
        pub enum ValueType {
            $($ident($t)),*
        }



        $(
            impl ToValue<$t> for ValueType {
                fn to_value(&self) -> Option<$t> {
                    if let ValueType::$ident(val) = self {
                        Some(val.clone())
                    } else {
                        None
                    }
                }
            }

            impl From<$t> for ValueType {
                fn from(val: $t) -> Self {
                    ValueType::$ident(val)
                }
            }
        )*

        paste::paste! {
            $(#[allow(dead_code)] type [<$ident Inner>] = $t;)*
            $(#[allow(dead_code)] type [<$trait Inner>] = [<$first_impl Inner>];)*
        }
    }
}

pub trait ToValue<T>: Sized {
    fn to_value(&self) -> Option<T>;
}

pub trait MaxBy: Sized {
    fn max_by<T, F: Fn(&Self) -> T>(self, other: Self, f: F) -> Self
    where
        T: PartialOrd,
    {
        if f(&self) > f(&other) {
            self
        } else {
            other
        }
    }
}

impl<T> MaxBy for T {}

macro_rules! templates {
    ($($ident:ident ($($input_name:ident: $input:ident),*$(,)?) ($($output_name:ident: $output:ident),*$(,)?) $name:literal),*$(,)?) => {
        #[derive(EnumIter, Copy, Clone, Serialize, Deserialize)]
        pub enum NodeTemplate {
            $($ident),*
        }
        impl NodeTemplate {
            #[allow(unused_mut, unused_assignments)]
            pub fn update_connection(&self, graph: &mut Graph, node_id: NodeId) {
                match self {
                    $(
                        NodeTemplate::$ident => {
                            let mut outputs = TRAIT_DEFAULT_ARRAY.clone();
                            let mut i = 0;
                            $(
                                {
                                    let input = graph.nodes[node_id].inputs[i].1;
                                    if let Some(output) = graph.connections.get(input) {
                                        let typ = graph.outputs[*output].typ;

                                        if !typ.eq(&DataType::$input) {
                                            graph.remove_connection(input);
                                        } else {
                                            const IS_TRAIT: bool = DataType::$input.is_trait();
                                            if IS_TRAIT {
                                                // Hack to get around compiler complaining about out of index
                                                let index = DataType::$input as usize * IS_TRAIT as usize;
                                                outputs[index] = typ.max_by(outputs[index], |typ| typ.derive_order(DataType::$input));
                                            }
                                        }

                                    }
                                    i += 1;
                                }
                            )*
                            i = 0;
                            $(
                                {
                                    const IS_TRAIT: bool = DataType::$output.is_trait();
                                    if IS_TRAIT {
                                        let index = DataType::$output as usize * IS_TRAIT as usize;
                                        let output = graph.nodes[node_id].outputs[i].1;
                                        let o = &mut graph.outputs[output];
                                        if o.typ as u32 != outputs[index] as u32 {
                                            o.typ = outputs[index];
                                            let connected = graph.connections.iter().filter_map(|(i, o)| if output == *o { Some(graph.inputs[i].node) } else { None}).collect::<HashSet<_>>();
                                            for node_id in connected {
                                                let node = &graph.nodes[node_id];
                                                let t = node.user_data.template.clone();
                                                t.update_connection(graph, node_id);
                                            }
                                        }
                                        i += 1;
                                    }
                                }
                            )*
                        }
                    )*
                }
            }
        }

        impl NodeTemplateTrait for NodeTemplate {
            type NodeData = NodeData;

            type DataType = DataType;

            type ValueType = ValueType;

            fn node_finder_label(&self) -> &str {
                match self {
                    $(NodeTemplate::$ident => $name),*,
                }
            }

            fn node_graph_label(&self) -> String {
                self.node_finder_label().to_string()
            }

            fn user_data(&self) -> Self::NodeData {
                NodeData {
                    template: self.clone(),
                }
            }

            fn build_node(&self, graph: &mut Graph, node_id: NodeId) {
                let input = |graph: &mut Graph, name: &str, typ: DataType| {
                    graph.add_input_param(
                        node_id,
                        name.to_string(),
                        typ,
                        typ.first_impl().with_default(),
                        InputParamKind::ConnectionOrConstant,
                        true,
                    );
                };
                let output = |graph: &mut Graph, name: &str, typ: DataType| {
                    graph.add_output_param(node_id, name.to_string(), typ);
                };
                match self {
                    $(
                        NodeTemplate::$ident => {
                            $(
                                input(graph, stringify!($input_name), DataType::$input);
                            )*
                            $(
                                output(graph, stringify!($output_name), DataType::$output);
                            )*
                        }
                    )*
                }
            }
        }
    }
}
#[derive(AsRefStr, EnumIter, PartialEq, Eq, Debug, Copy, Clone, Serialize, Deserialize)]
#[repr(u8)]
pub enum BlockKind {
    Air = 0x00, // Air counts as a fluid
    Water = 0x01,
    Rock = 0x10,
    WeakRock = 0x11,
    Lava = 0x12,
    GlowingRock = 0x13,
    GlowingWeakRock = 0x14,
    Snow = 0x21,
    Earth = 0x30,
    Sand = 0x31,
    Wood = 0x40,
    Leaves = 0x41,
    GlowingMushroom = 0x42,
    Ice = 0x43,
    Misc = 0xFE,
}

impl BlockKind {
    #[inline]
    pub const fn is_air(&self) -> bool {
        matches!(self, BlockKind::Air)
    }

    /// Determine whether the block kind is a gas or a liquid. This does not
    /// consider any sprites that may occupy the block (the definition of
    /// fluid is 'a substance that deforms to fit containers')
    #[inline]
    pub const fn is_fluid(&self) -> bool {
        *self as u8 & 0xF0 == 0x00
    }

    /// Determine whether the block is filled (i.e: fully solid). Right now,
    /// this is the opposite of being a fluid.
    #[inline]
    pub const fn is_filled(&self) -> bool {
        !self.is_fluid()
    }
}

impl Default for BlockKind {
    fn default() -> Self {
        BlockKind::Air
    }
}

#[derive(AsRefStr, EnumIter, PartialEq, Eq, Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Dir {
    X,
    Y,
    NegX,
    NegY,
}

impl From<Dir> for Vec2<i32> {
    fn from(dir: Dir) -> Self {
        match dir {
            Dir::X => Vec2::new(1, 0),
            Dir::Y => Vec2::new(0, 1),
            Dir::NegX => Vec2::new(-1, 0),
            Dir::NegY => Vec2::new(0, -1),
        }
    }
}

impl Default for Dir {
    fn default() -> Self {
        Dir::X
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Primitive {
    Empty,
    Aabb(Aabb<i32>),
    Sphere(Vec3<f32>, f32),
    Cylinder {
        center: Vec2<f32>,
        radius: f32,
        zmin: i32,
        zmax: i32,
    },
    Ramp(Aabb<i32>, Dir),

    Union(Vec<Primitive>),
    Intersection(Vec<Primitive>),
    Difference(Box<Primitive>, Box<Primitive>),
    Translate(Box<Primitive>, Vec3<i32>),
}

impl Primitive {
    pub fn bounds(&self) -> Option<Aabb<i32>> {
        let aabb = match self {
            Primitive::Empty => None,
            Primitive::Aabb(aabb) => Some(*aabb),
            Primitive::Sphere(center, radius) => Some(Aabb {
                min: (center - radius).as_(),
                max: (center + radius).as_(),
            }),
            Primitive::Cylinder {
                center,
                radius,
                zmin,
                zmax,
            } => Some(Aabb {
                min: (center - radius).as_().with_z(*zmin),
                max: (center + radius).ceil().as_().with_z(*zmax),
            }),
            Primitive::Ramp(aabb, _) => Some(*aabb),
            Primitive::Union(primitives) => primitives
                .iter()
                .filter_map(|p| p.bounds())
                .reduce(|a, b| a.union(b)),
            Primitive::Intersection(primitives) => primitives
                .iter()
                .filter_map(|p| p.bounds())
                .try_reduce(|a, b| {
                    let aabb = a.intersection(b);
                    (aabb.size().reduce_min() > 0).then_some(aabb)
                })
                .flatten(),
            Primitive::Difference(primitive, _) => primitive.bounds(),
            Primitive::Translate(primitive, trans) => primitive.bounds().map(|aabb| Aabb {
                min: aabb.min + trans,
                max: aabb.max + trans,
            }),
        };
        aabb.filter(|aabb| aabb.size().reduce_min() > 0)
    }

    pub fn contains_at(&self, p: Vec3<i32>) -> bool {
        match self {
            Primitive::Empty => false,
            Primitive::Aabb(aabb) => aabb.contains_point(p),
            Primitive::Sphere(center, radius) => center.distance_squared(p.as_()) < radius.powi(2),
            Primitive::Cylinder {
                center,
                radius,
                zmin,
                zmax,
            } => {
                (*zmin..=*zmax).contains(&p.z)
                    && center.distance_squared(p.xy().as_()) < radius.powi(2)
            }
            Primitive::Ramp(aabb, dir) => {
                let inset = aabb.size().reduce_min();
                let inner = match dir {
                    Dir::X => Aabr {
                        min: Vec2::new(aabb.min.x - 1, aabb.min.y),
                        max: Vec2::new(aabb.max.x, aabb.max.y),
                    },
                    Dir::NegX => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y),
                        max: Vec2::new(aabb.max.x, aabb.max.y),
                    },
                    Dir::Y => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y - 1),
                        max: Vec2::new(aabb.max.x, aabb.max.y),
                    },
                    Dir::NegY => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y),
                        max: Vec2::new(aabb.max.x, aabb.max.y),
                    },
                };
                aabb.contains_point(p)
                    && (inner.projected_point(p.xy()) - p.xy())
                        .map(|e| e.abs())
                        .reduce_max() as f32
                        / (inset as f32)
                        < 1.0 - ((p.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32
            }
            Primitive::Union(primitives) => primitives.iter().any(|prim| prim.contains_at(p)),
            Primitive::Intersection(primitives) => {
                primitives.iter().all(|prim| prim.contains_at(p))
            }
            Primitive::Difference(a, b) => a.contains_at(p) && !b.contains_at(p),
            Primitive::Translate(primitive, trans) => primitive.contains_at(p - trans),
        }
    }
}

impl Default for Primitive {
    fn default() -> Self {
        Primitive::Empty
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Fill {
    Block(BlockKind, Rgb<u8>),
    Brick(BlockKind, Rgb<u8>, u8),
}
fn randomize(seed: u32, pos: Vec3<i32>) -> u32 {
    let pos = pos.map(|e| u32::from_le_bytes(e.to_le_bytes()));

    let mut a = seed;
    a = (a ^ 61) ^ (a >> 16);
    a = a.wrapping_add(a << 3);
    a ^= pos.x;
    a ^= a >> 4;
    a = a.wrapping_mul(0x27d4eb2d);
    a ^= a >> 15;
    a ^= pos.y;
    a = (a ^ 61) ^ (a >> 16);
    a = a.wrapping_add(a << 3);
    a ^= a >> 4;
    a ^= pos.z;
    a = a.wrapping_mul(0x27d4eb2d);
    a ^= a >> 15;
    a
}

#[derive(Default, Clone, Copy)]
pub struct Block(BlockKind, Rgb<u8>);

impl IsOpaque for Block {
    fn is_opaque(&self) -> bool {
        self.0.is_filled()
    }
}
impl IsEmpty for Block {
    fn is_empty(&self) -> bool {
        self.0.is_air()
    }
}
impl MergeVoxel for Block {
    type VoxelValue = Rgb<u8>;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        self.1
    }
}

impl Fill {
    pub fn sample_at(&self, p: Vec3<i32>) -> Block {
        match self {
            Fill::Block(block_type, color) => Block(*block_type, *color),
            Fill::Brick(block_type, color, range) => Block(*block_type, {
                let offset = (randomize(5441, (p + Vec3::new(p.y, 0, p.y)) / Vec3::new(2, 1, 2))
                    % *range as u32) as u8;
                color.map(|c| c.saturating_add(offset))
            }),
        }
    }
}

impl Default for Fill {
    fn default() -> Self {
        Fill::Block(BlockKind::default(), Rgb::default())
    }
}

types! {
    Dir(Dir) @GOLD,
    I32(i32) @LIGHT_BLUE,
    IVec2(Vec2<i32>) @BLUE,
    IVec3(Vec3<i32>) @BLUE,
    IMat3(Mat3<i32>) @BLUE,
    Aabb(Aabb<i32>) @DARK_BLUE,
    Aabr(Aabr<i32>) @DARK_BLUE,

    F32(f32) @LIGHT_YELLOW,
    Vec2(Vec2<f32>) @YELLOW,
    Vec3(Vec3<f32>) @YELLOW,

    BlockType(BlockKind) @LIGHT_RED,
    Color(Rgb<u8>) @RED,
    Primitive(Primitive) @GREEN,
    Fill(Fill) @GREEN;
    IVec[I32, IVec2, IVec3],
    Vec[F32, Vec2, Vec3],
}

pub fn vec2_ui<T: Numeric>(ui: &mut egui::Ui, value: &mut Vec2<T>) -> egui::Response {
    ui.label("x");
    let x = ui.add(egui::DragValue::new(&mut value.x));
    ui.label("y");
    let y = ui.add(egui::DragValue::new(&mut value.y));
    x.union(y)
}
pub fn vec3_ui<T: Numeric>(ui: &mut egui::Ui, value: &mut Vec3<T>) -> egui::Response {
    ui.label("x");
    let x = ui.add(egui::DragValue::new(&mut value.x));
    ui.label("y");
    let y = ui.add(egui::DragValue::new(&mut value.y));
    ui.label("z");
    let z = ui.add(egui::DragValue::new(&mut value.z));
    x.union(y).union(z)
}
pub fn aabr_ui<T: Numeric>(ui: &mut egui::Ui, value: &mut Aabr<T>) -> egui::Response {
    let min = ui
        .horizontal(|ui| {
            ui.label("min");
            vec2_ui(ui, &mut value.min)
        })
        .inner;
    let max = ui
        .horizontal(|ui| {
            ui.label("max");
            vec2_ui(ui, &mut value.max)
        })
        .inner;
    min.union(max)
}
impl WidgetValueTrait for ValueType {
    type Response = self::Response;
    fn value_widget(&mut self, param_name: &str, ui: &mut egui::Ui) -> Vec<Self::Response> {
        let response = match self {
            ValueType::I32(value) => {
                ui.horizontal(|ui| {
                    ui.label(param_name);
                    ui.add(egui::DragValue::new(value))
                })
                .inner
            }
            ValueType::IVec2(value) => {
                ui.label(param_name);
                ui.horizontal(|ui| vec2_ui(ui, value)).inner
            }
            ValueType::IVec3(value) => {
                ui.label(param_name);
                ui.horizontal(|ui| vec3_ui(ui, value)).inner
            }
            ValueType::IMat3(value) => {
                ui.label(param_name);
                ui.label("x");
                let x = ui
                    .horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut value.cols.x.x));
                        ui.add(egui::DragValue::new(&mut value.cols.x.y));
                        ui.add(egui::DragValue::new(&mut value.cols.x.z))
                    })
                    .inner;
                ui.label("y");
                let y = ui
                    .horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut value.cols.y.x));
                        ui.add(egui::DragValue::new(&mut value.cols.y.y));
                        ui.add(egui::DragValue::new(&mut value.cols.y.z))
                    })
                    .inner;
                ui.label("z");
                let z = ui
                    .horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut value.cols.z.x));
                        ui.add(egui::DragValue::new(&mut value.cols.z.y));
                        ui.add(egui::DragValue::new(&mut value.cols.z.z))
                    })
                    .inner;
                x.union(y).union(z)
            }
            ValueType::Aabb(value) => {
                ui.label(param_name);
                let min = ui
                    .horizontal(|ui| {
                        ui.label("min");
                        vec3_ui(ui, &mut value.min)
                    })
                    .inner;
                let max = ui
                    .horizontal(|ui| {
                        ui.label("max");
                        vec3_ui(ui, &mut value.max)
                    })
                    .inner;
                min.union(max)
            }
            ValueType::Aabr(value) => {
                ui.label(param_name);
                aabr_ui(ui, value)
            }

            ValueType::F32(value) => {
                ui.horizontal(|ui| {
                    ui.label(param_name);
                    ui.add(egui::DragValue::new(value))
                })
                .inner
            }
            ValueType::Vec2(value) => {
                ui.label(param_name);
                ui.horizontal(|ui| vec2_ui(ui, value)).inner
            }
            ValueType::Vec3(value) => {
                ui.label(param_name);
                ui.horizontal(|ui| vec3_ui(ui, value)).inner
            }
            ValueType::BlockType(value) => {
                ui.label(param_name);
                egui::ComboBox::new("block type", "")
                    .selected_text(value.as_ref())
                    .show_ui(ui, |ui| {
                        BlockKind::iter().for_each(|typ| {
                            ui.selectable_value(value, typ, typ.as_ref());
                        });
                    })
                    .response
            }
            ValueType::Color(value) => {
                ui.label(param_name);
                ui.horizontal(|ui| {
                    let mut color = [
                        value.r as f32 / 255.0,
                        value.g as f32 / 255.0,
                        value.b as f32 / 255.0,
                    ];
                    let res = ui.color_edit_button_rgb(&mut color);
                    *value = Rgb::new(color[0] * 255.0, color[1] * 255.0, color[2] * 255.0).as_();
                    res
                })
                .inner
            }
            ValueType::Dir(value) => {
                ui.label(param_name);
                egui::ComboBox::new("dir", "")
                    .selected_text(value.as_ref())
                    .show_ui(ui, |ui| {
                        Dir::iter().for_each(|typ| {
                            ui.selectable_value(value, typ, typ.as_ref());
                        });
                    })
                    .response
            }
            ValueType::Primitive(_) | ValueType::Fill(_) => {
                ui.label(param_name);
                return Vec::new();
            }
        };
        if response.changed() {
            vec![self::Response::Changed]
        } else {
            Vec::new()
        }
    }
}

templates! {
    MakeDir(in: Dir)(out: Dir) "new dir",
    MakeI32(in: I32)(out: I32) "new i32",
    MakeIVec2(x: I32, y: I32)(out: IVec2) "new ivec2",
    MakeIVec3(x: I32, y: I32, z: I32)(out: IVec3) "new ivec3",
    MakeIMat3(x: IVec3, y: IVec3, z: IVec3)(out: IMat3) "new mat3x3",
    MakeAabb(min: IVec3, max: IVec3)(out: Aabb) "new aabb",
    MakeAabr(min: IVec2, max: IVec2)(out: Aabr) "new aabr",
    MakeF32(in: F32)(out: F32) "new f32",
    MakeVec2(x: F32, y: F32)(out: Vec2) "new vec2",
    MakeVec3(x: F32, y: F32, z: F32)(out: Vec3) "new vec3",
    MakeBlockType(in: BlockType)(out: BlockType) "new block type",
    MakeColor(in: Color)(out: Color) "new color",
    // Float
    AddF(a: Vec, b: Vec)(out: Vec) "add scalar",
    SubF(a: Vec, b: Vec)(out: Vec) "sub scalar",
    MulF(a: Vec, b: Vec)(out: Vec) "mul scalar",
    DivF(a: Vec, b: Vec)(out: Vec) "div scalar",
    // Integer
    AddI(a: IVec, b: IVec)(out: IVec) "add Integer",
    SubI(a: IVec, b: IVec)(out: IVec) "sub Integer",
    MulI(a: IVec, b: IVec)(out: IVec) "mul integer",
    DivI(a: IVec, b: IVec)(out: IVec) "div integer",

    // Vec
    Vec2Components(vec: Vec2)(x: F32, y: F32) "vec2 components",
    Vec3Components(vec: Vec3)(x: F32, y: F32, z: F32) "vec3 components",

    // IVec
    IVec2Components(vec: IVec2)(x: I32, y: I32) "ivec2 components",
    IVec3Components(vec: IVec3)(x: I32, y: I32, z: I32) "ivec3 components",

    // Aabr
    AabrToAabb(aabr: Aabr, min: I32, max: I32)(out: Aabb) "aabr to aabb",
    AabrComponents(aabr: Aabr)(min: IVec3, max: IVec3) "aabr components",
    // Aabb
    AabbComponents(aabr: Aabb)(min: IVec3, max: IVec3) "aabb components",
    AabbToAabr(aabr: Aabb)(out: Aabr, min: I32, max: I32) "aabb to aabr",

    // Fills
    FillBlock(block: BlockType, color: Color)(out: Fill) "block fill",
    FillEmpty()(out: Fill) "empty fill",
    FillBrick(block: BlockType, color: Color, offset: I32)(out: Fill) "fill brick",

    FillPrimtive(prim: Primitive, fill: Fill)() "fill",

    // Primitives
    PrimAabb(aabb: Aabb)(out: Primitive) "aabb primitive",
    PrimSphere(center: Vec3, radius: F32)(out: Primitive) "sphere primitive",
    PrimSphereAabb(aabb: Aabb)(out: Primitive) "sphere primitive from aabb",
    PrimCylinderAabb(aabb: Aabb)(out: Primitive) "cylinder primitive from aabb",
    PrimRamp(aabb: Aabb, dir: Dir)(out: Primitive) "ramp primitive",

    Union(prim: Primitive)(out: Primitive) "union",
    Intersection(prim: Primitive)(out: Primitive) "intersection",
    Difference(a: Primitive, b: Primitive)(out: Primitive) "difference",

    Translate(prim: Primitive, offset: IVec3)(out: Primitive) "translate",

    // Parameters
    Alt()(out: I32) "altitude",
    Bounds()(out: Aabr) "bounds",
}
