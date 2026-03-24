"""
Blender headless script: import GLB -> select all meshes -> apply transforms -> join -> export GLB.
Replicates Blender's manual Ctrl+A (Apply All) + Ctrl+J (Join) workflow.

Usage: blender -b --python _blender_join.py -- input.glb output.glb
"""
import bpy
import sys


def main():
    argv = sys.argv
    args = argv[argv.index("--") + 1:]
    input_path = args[0]
    output_path = args[1]

    # Clear default scene objects (cube, camera, light)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=input_path)

    # Select only mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not mesh_objects:
        print("ERROR: No mesh objects found after import", file=sys.stderr)
        sys.exit(1)

    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    # Apply all transforms (location, rotation, scale) to mesh data
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Join all selected mesh objects into one
    if len(mesh_objects) > 1:
        bpy.ops.object.join()

    # Export as GLB
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        check_existing=False,
    )

    print(f"OK: joined {len(mesh_objects)} mesh objects -> {output_path}")


if __name__ == '__main__':
    main()
