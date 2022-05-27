#![feature(
    split_array,
    iterator_try_collect,
    drain_filter,
    let_chains,
    iterator_try_reduce
)]
use bevy::prelude::*;
use bevy_flycam::PlayerPlugin;

mod node;

fn main() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(bevy_egui::EguiPlugin)
        .add_plugin(PlayerPlugin)
        .add_startup_system(startup_system)
        .add_system(node::ui)
        .run();
}

fn startup_system(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    let mat = materials.add(StandardMaterial {
        base_color: Color::GRAY,
        ..default()
    });
    commands.insert_resource(mat);
    commands.spawn_bundle(DirectionalLightBundle {
        transform: Transform::from_translation(Vec3::new(4.0, 10.0, 4.0))
            .looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.spawn_bundle(DirectionalLightBundle {
        transform: Transform::from_translation(Vec3::new(-4.0, -10.0, -4.0))
            .looking_at(Vec3::ZERO, Vec3::Y),
        directional_light: DirectionalLight {
            illuminance: 20000.0,
            ..default()
        },
        ..default()
    });
}
