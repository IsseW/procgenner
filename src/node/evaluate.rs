use bevy::utils::HashMap;
use egui_node_graph::{InputId, Node, OutputId};
use vek::*;

use crate::node::template::{BlockKind, Dir, Fill, Primitive};

use super::{
    template::{NodeTemplate, ToValue, ValueType},
    Graph, NodeData, Structure,
};

pub struct Ctx<'a> {
    pub graph: &'a Graph,
    pub structure: &'a Structure,
    pub cached: &'a mut HashMap<OutputId, ValueType>,
}

impl NodeTemplate {
    fn evaluate(&self, node: &Node<NodeData>, ctx: &mut Ctx) -> Option<Vec<ValueType>> {
        let input = |i: usize| node.inputs[i].1;
        fn checked<T>(input: InputId, ctx: &mut Ctx) -> Option<ValueType>
        where
            ValueType: ToValue<T> + From<T>,
        {
            Some(ValueType::from(eval::<T>(input, ctx)?))
        }
        use NodeTemplate::*;
        Some(match self {
            MakeDir => {
                vec![checked::<Dir>(input(0), ctx)?]
            }
            MakeI32 => {
                vec![checked::<i32>(input(0), ctx)?]
            }
            MakeIVec2 => {
                let x = eval::<i32>(input(0), ctx)?;
                let y = eval::<i32>(input(1), ctx)?;
                vec![ValueType::from(Vec2::new(x, y))]
            }
            MakeIVec3 => {
                let x = eval::<i32>(input(0), ctx)?;
                let y = eval::<i32>(input(1), ctx)?;
                let z = eval::<i32>(input(2), ctx)?;
                vec![ValueType::from(Vec3::new(x, y, z))]
            }
            MakeIMat3 => {
                let x = eval::<Vec3<i32>>(input(0), ctx)?;
                let y = eval::<Vec3<i32>>(input(1), ctx)?;
                let z = eval::<Vec3<i32>>(input(2), ctx)?;
                vec![ValueType::from(Mat3 {
                    cols: Vec3::new(x, y, z),
                })]
            }
            MakeAabb => {
                let min = eval::<Vec3<i32>>(input(0), ctx)?;
                let max = eval::<Vec3<i32>>(input(1), ctx)?;
                vec![ValueType::from(Aabb { min, max })]
            }
            MakeAabr => {
                let min = eval::<Vec2<i32>>(input(0), ctx)?;
                let max = eval::<Vec2<i32>>(input(1), ctx)?;
                vec![ValueType::from(Aabr { min, max })]
            }
            MakeF32 => {
                vec![checked::<f32>(input(0), ctx)?]
            }
            MakeVec2 => {
                let x = eval::<f32>(input(0), ctx)?;
                let y = eval::<f32>(input(1), ctx)?;
                vec![ValueType::from(Vec2::new(x, y))]
            }
            MakeVec3 => {
                let x = eval::<f32>(input(0), ctx)?;
                let y = eval::<f32>(input(1), ctx)?;
                let z = eval::<f32>(input(1), ctx)?;
                vec![ValueType::from(Vec3::new(x, y, z))]
            }
            MakeBlockType => {
                vec![checked::<BlockKind>(input(0), ctx)?]
            }
            MakeColor => {
                vec![checked::<Rgb<u8>>(input(0), ctx)?]
            }
            AddF => {
                let a = eval_raw(input(0), ctx)?;
                let b = eval_raw(input(1), ctx)?;
                match (a, b) {
                    (ValueType::F32(a), ValueType::F32(b)) => vec![ValueType::F32(a + b)],
                    (ValueType::Vec2(a), ValueType::Vec2(b)) => vec![ValueType::Vec2(a + b)],
                    (ValueType::Vec2(a), ValueType::F32(b)) => vec![ValueType::Vec2(a + b)],
                    (ValueType::Vec3(a), ValueType::Vec3(b)) => vec![ValueType::Vec3(a + b)],
                    (ValueType::Vec3(a), ValueType::F32(b)) => vec![ValueType::Vec3(a + b)],
                    (ValueType::Vec3(a), ValueType::Vec2(b)) => {
                        vec![ValueType::Vec3(a + b.with_z(0.0))]
                    }
                    _ => return None,
                }
            }
            SubF => {
                let a = eval_raw(input(0), ctx)?;
                let b = eval_raw(input(1), ctx)?;
                match (a, b) {
                    (ValueType::F32(a), ValueType::F32(b)) => vec![ValueType::F32(a - b)],
                    (ValueType::Vec2(a), ValueType::Vec2(b)) => vec![ValueType::Vec2(a - b)],
                    (ValueType::Vec2(a), ValueType::F32(b)) => vec![ValueType::Vec2(a - b)],
                    (ValueType::Vec3(a), ValueType::Vec3(b)) => vec![ValueType::Vec3(a - b)],
                    (ValueType::Vec3(a), ValueType::F32(b)) => vec![ValueType::Vec3(a - b)],
                    (ValueType::Vec3(a), ValueType::Vec2(b)) => {
                        vec![ValueType::Vec3(a - b.with_z(0.0))]
                    }
                    _ => return None,
                }
            }
            MulF => {
                let a = eval_raw(input(0), ctx)?;
                let b = eval_raw(input(1), ctx)?;
                match (a, b) {
                    (ValueType::F32(a), ValueType::F32(b)) => vec![ValueType::F32(a * b)],
                    (ValueType::Vec2(a), ValueType::Vec2(b)) => vec![ValueType::Vec2(a * b)],
                    (ValueType::Vec2(a), ValueType::F32(b)) => vec![ValueType::Vec2(a * b)],
                    (ValueType::Vec3(a), ValueType::Vec3(b)) => vec![ValueType::Vec3(a * b)],
                    (ValueType::Vec3(a), ValueType::F32(b)) => vec![ValueType::Vec3(a * b)],
                    (ValueType::Vec3(a), ValueType::Vec2(b)) => {
                        vec![ValueType::Vec3(a * b.with_z(1.0))]
                    }
                    _ => return None,
                }
            }
            DivF => {
                let a = eval_raw(input(0), ctx)?;
                let b = eval_raw(input(1), ctx)?;
                match (a, b) {
                    (ValueType::F32(a), ValueType::F32(b)) => vec![ValueType::F32(a / b)],
                    (ValueType::Vec2(a), ValueType::Vec2(b)) => vec![ValueType::Vec2(a / b)],
                    (ValueType::Vec2(a), ValueType::F32(b)) => vec![ValueType::Vec2(a / b)],
                    (ValueType::Vec3(a), ValueType::Vec3(b)) => vec![ValueType::Vec3(a / b)],
                    (ValueType::Vec3(a), ValueType::F32(b)) => vec![ValueType::Vec3(a / b)],
                    (ValueType::Vec3(a), ValueType::Vec2(b)) => {
                        vec![ValueType::Vec3(a / b.with_z(1.0))]
                    }
                    _ => return None,
                }
            }
            AddI => {
                let a = eval_raw(input(0), ctx)?;
                let b = eval_raw(input(1), ctx)?;
                match (a, b) {
                    (ValueType::I32(a), ValueType::I32(b)) => vec![ValueType::I32(a + b)],
                    (ValueType::IVec2(a), ValueType::IVec2(b)) => vec![ValueType::IVec2(a + b)],
                    (ValueType::IVec2(a), ValueType::I32(b)) => vec![ValueType::IVec2(a + b)],
                    (ValueType::IVec3(a), ValueType::IVec3(b)) => vec![ValueType::IVec3(a + b)],
                    (ValueType::IVec3(a), ValueType::I32(b)) => vec![ValueType::IVec3(a + b)],
                    (ValueType::IVec3(a), ValueType::IVec2(b)) => {
                        vec![ValueType::IVec3(a + b.with_z(0))]
                    }
                    _ => return None,
                }
            }
            SubI => {
                let a = eval_raw(input(0), ctx)?;
                let b = eval_raw(input(1), ctx)?;
                match (a, b) {
                    (ValueType::I32(a), ValueType::I32(b)) => vec![ValueType::I32(a - b)],
                    (ValueType::IVec2(a), ValueType::IVec2(b)) => vec![ValueType::IVec2(a - b)],
                    (ValueType::IVec2(a), ValueType::I32(b)) => vec![ValueType::IVec2(a - b)],
                    (ValueType::IVec3(a), ValueType::IVec3(b)) => vec![ValueType::IVec3(a - b)],
                    (ValueType::IVec3(a), ValueType::I32(b)) => vec![ValueType::IVec3(a - b)],
                    (ValueType::IVec3(a), ValueType::IVec2(b)) => {
                        vec![ValueType::IVec3(a - b.with_z(0))]
                    }
                    _ => return None,
                }
            }
            MulI => {
                let a = eval_raw(input(0), ctx)?;
                let b = eval_raw(input(1), ctx)?;
                match (a, b) {
                    (ValueType::I32(a), ValueType::I32(b)) => vec![ValueType::I32(a * b)],
                    (ValueType::IVec2(a), ValueType::IVec2(b)) => vec![ValueType::IVec2(a * b)],
                    (ValueType::IVec2(a), ValueType::I32(b)) => vec![ValueType::IVec2(a * b)],
                    (ValueType::IVec3(a), ValueType::IVec3(b)) => vec![ValueType::IVec3(a * b)],
                    (ValueType::IVec3(a), ValueType::I32(b)) => vec![ValueType::IVec3(a * b)],
                    (ValueType::IVec3(a), ValueType::IVec2(b)) => {
                        vec![ValueType::IVec3(a * b.with_z(1))]
                    }
                    _ => return None,
                }
            }
            DivI => {
                let a = eval_raw(input(0), ctx)?;
                let b = eval_raw(input(1), ctx)?;
                match (a, b) {
                    (ValueType::I32(a), ValueType::I32(b)) => vec![ValueType::I32(a / b)],
                    (ValueType::IVec2(a), ValueType::IVec2(b)) => vec![ValueType::IVec2(a / b)],
                    (ValueType::IVec2(a), ValueType::I32(b)) => vec![ValueType::IVec2(a / b)],
                    (ValueType::IVec3(a), ValueType::IVec3(b)) => vec![ValueType::IVec3(a / b)],
                    (ValueType::IVec3(a), ValueType::I32(b)) => vec![ValueType::IVec3(a / b)],
                    (ValueType::IVec3(a), ValueType::IVec2(b)) => {
                        vec![ValueType::IVec3(a / b.with_z(1))]
                    }
                    _ => return None,
                }
            }
            Vec2Components => {
                let v = eval::<Vec2<f32>>(input(0), ctx)?;
                vec![ValueType::from(v.x), ValueType::from(v.y)]
            }
            Vec3Components => {
                let v = eval::<Vec3<f32>>(input(0), ctx)?;
                vec![
                    ValueType::from(v.x),
                    ValueType::from(v.y),
                    ValueType::from(v.z),
                ]
            }
            IVec2Components => {
                let v = eval::<Vec2<i32>>(input(0), ctx)?;
                vec![ValueType::from(v.x), ValueType::from(v.y)]
            }
            IVec3Components => {
                let v = eval::<Vec3<i32>>(input(0), ctx)?;
                vec![
                    ValueType::from(v.x),
                    ValueType::from(v.y),
                    ValueType::from(v.z),
                ]
            }
            AabrToAabb => {
                let aabr = eval::<Aabr<i32>>(input(0), ctx)?;
                let min = eval::<i32>(input(1), ctx)?;
                let max = eval::<i32>(input(2), ctx)?;
                vec![ValueType::from(Aabb {
                    min: aabr.min.with_z(min),
                    max: aabr.max.with_z(max),
                })]
            }
            AabrComponents => {
                let aabr = eval::<Aabr<i32>>(input(0), ctx)?;
                vec![ValueType::from(aabr.min), ValueType::from(aabr.max)]
            }
            AabbComponents => {
                let aabb = eval::<Aabb<i32>>(input(0), ctx)?;
                vec![ValueType::from(aabb.min), ValueType::from(aabb.max)]
            }
            AabbToAabr => {
                let aabb = eval::<Aabb<i32>>(input(0), ctx)?;
                let aabr = Aabr {
                    min: aabb.min.xy(),
                    max: aabb.max.xy(),
                };
                vec![
                    ValueType::from(aabr),
                    ValueType::from(aabb.min.z),
                    ValueType::from(aabb.max.z),
                ]
            }
            FillBlock => {
                let block = eval::<BlockKind>(input(0), ctx)?;
                let color = eval::<Rgb<u8>>(input(1), ctx)?;
                vec![ValueType::from(Fill::Block(block, color))]
            }
            FillEmpty => {
                vec![ValueType::from(Fill::default())]
            }
            FillBrick => {
                let block = eval::<BlockKind>(input(0), ctx)?;
                let color = eval::<Rgb<u8>>(input(1), ctx)?;
                let variance = eval::<i32>(input(2), ctx)?;
                vec![ValueType::from(Fill::Brick(block, color, variance as u8))]
            }
            FillPrimtive => return None,
            PrimAabb => {
                let aabb = eval::<Aabb<i32>>(input(0), ctx)?;
                vec![ValueType::from(Primitive::Aabb(aabb))]
            }
            PrimSphere => {
                let center = eval::<Vec3<f32>>(input(0), ctx)?;
                let radius = eval::<f32>(input(1), ctx)?;
                vec![ValueType::from(Primitive::Sphere(center, radius))]
            }
            PrimSphereAabb => {
                let aabb = eval::<Aabb<i32>>(input(0), ctx)?;
                let center = aabb.as_::<f32>().center();
                let radius = aabb.size().reduce_min() as f32 / 2.0;
                vec![ValueType::from(Primitive::Sphere(center, radius))]
            }
            PrimCylinderAabb => {
                let aabb = eval::<Aabb<i32>>(input(0), ctx)?;
                let aabr = Aabr {
                    min: aabb.min.xy(),
                    max: aabb.max.xy(),
                };
                let center = aabr.as_::<f32>().center();
                let radius = aabr.size().reduce_min() as f32 / 2.0;
                let zmin = aabb.min.z;
                let zmax = aabb.max.z;
                vec![ValueType::from(Primitive::Cylinder {
                    center,
                    radius,
                    zmin,
                    zmax,
                })]
            }
            PrimRamp => {
                let aabb = eval::<Aabb<i32>>(input(0), ctx)?;
                let dir = eval::<Dir>(input(1), ctx)?;
                vec![ValueType::from(Primitive::Ramp(aabb, dir))]
            }
            Union => {
                let prims = node
                    .inputs
                    .iter()
                    .map(|i| eval::<Primitive>(i.1, ctx))
                    .try_collect()?;
                vec![ValueType::from(Primitive::Union(prims))]
            }
            Intersection => {
                let prims = node
                    .inputs
                    .iter()
                    .map(|i| eval::<Primitive>(i.1, ctx))
                    .try_collect()?;
                vec![ValueType::from(Primitive::Intersection(prims))]
            }
            Difference => {
                let a = eval::<Primitive>(input(0), ctx)?;
                let b = eval::<Primitive>(input(1), ctx)?;
                vec![ValueType::from(Primitive::Difference(
                    Box::new(a),
                    Box::new(b),
                ))]
            }
            Translate => {
                let prim = eval::<Primitive>(input(0), ctx)?;
                let offset = eval::<Vec3<i32>>(input(1), ctx)?;
                vec![ValueType::from(Primitive::Translate(
                    Box::new(prim),
                    offset,
                ))]
            }
            Alt => {
                vec![ValueType::from(ctx.structure.alt)]
            }
            Bounds => {
                vec![ValueType::from(ctx.structure.bounds)]
            }
        })
    }
}

fn eval_output(output: OutputId, ctx: &mut Ctx) -> Option<ValueType> {
    ctx.cached.get(&output).cloned().or_else(|| {
        let out = &ctx.graph.outputs[output];
        let node = &ctx.graph.nodes[out.node];
        let outputs = node.user_data.template.evaluate(node, ctx)?;
        let mut out = None;
        node.outputs
            .iter()
            .zip(outputs.into_iter())
            .for_each(|((_, id), value)| {
                if *id == output {
                    out = Some(value.clone());
                }
                ctx.cached.insert(*id, value);
            });
        out
    })
}

pub fn eval<T>(input: InputId, ctx: &mut Ctx) -> Option<T>
where
    ValueType: ToValue<T>,
{
    eval_raw(input, ctx).and_then(|v| v.to_value())
}

pub fn eval_raw(input: InputId, ctx: &mut Ctx) -> Option<ValueType> {
    if let Some(output) = ctx.graph.connections.get(input) {
        eval_output(*output, ctx)
    } else {
        Some(ctx.graph.inputs[input].value.clone())
    }
}
