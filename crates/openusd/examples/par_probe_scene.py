"""Generates a production-shaped synthetic set for composition benchmarks.

shot.usda (root) sublayers set.usda and anim.usda. set.usda places
groups x instances prop references under /World/Set; every prop asset
inherits a class from the root, carries a `lod` variant set and payloads its
geometry hierarchy from a separate file. Nothing is instanceable unless
--instanceable is given, so every prim composes on its own.

usage: par_probe_scene.py OUT_DIR GROUPS INSTANCES MESHES [PROPS] [--instanceable]
"""

import os
import sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    instanceable = "--instanceable" in sys.argv
    out, groups, instances, meshes = args[0], int(args[1]), int(args[2]), int(args[3])
    props = int(args[4]) if len(args) > 4 else 16
    os.makedirs(os.path.join(out, "assets"), exist_ok=True)

    for k in range(props):
        with open(os.path.join(out, "assets", f"prop_{k}_geom.usda"), "w") as f:
            f.write('#usda 1.0\n(\n    defaultPrim = "Prop"\n)\n\n')
            f.write('def Xform "Prop"\n{\n    def Scope "Geom"\n    {\n')
            for m in range(meshes):
                f.write(f'        def Mesh "Mesh_{m}" (\n            inherits = </_class_Mesh>\n        )\n        {{\n')
                f.write("            int[] faceVertexCounts = [4, 4]\n")
                f.write("            int[] faceVertexIndices = [0, 1, 2, 3, 2, 3, 4, 5]\n")
                f.write("            point3f[] points = [(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,2,0), (1,2,0)]\n")
                f.write('            uniform token subdivisionScheme = "none"\n')
                f.write(f"            double3 xformOp:translate = ({m}, 0, 0)\n")
                f.write('            uniform token[] xformOpOrder = ["xformOp:translate"]\n')
                f.write("        }\n")
            f.write("    }\n}\n")
        with open(os.path.join(out, "assets", f"prop_{k}.usda"), "w") as f:
            f.write('#usda 1.0\n(\n    defaultPrim = "Prop"\n)\n\n')
            f.write('def Xform "Prop" (\n    kind = "component"\n    inherits = </_class_Prop>\n')
            f.write(f"    payload = @./prop_{k}_geom.usda@\n")
            f.write('    variants = {\n        string lod = "high"\n    }\n')
            f.write('    prepend variantSets = "lod"\n)\n{\n')
            f.write(f'    custom string propName = "prop_{k}"\n')
            f.write('    variantSet "lod" = {\n')
            f.write('        "high" {\n            over "Geom" { token visibility = "inherited" }\n        }\n')
            f.write('        "low" {\n            over "Geom" { token visibility = "invisible" }\n        }\n')
            f.write("    }\n}\n\n")
            f.write('class "_class_Prop"\n{\n    custom int classOpinion = 0\n}\n\n')
            f.write('class "_class_Mesh"\n{\n    uniform bool doubleSided = 0\n}\n')

    with open(os.path.join(out, "set.usda"), "w") as f:
        f.write('#usda 1.0\n(\n    defaultPrim = "World"\n)\n\n')
        f.write('def Xform "World"\n{\n    def Xform "Set"\n    {\n')
        for g in range(groups):
            f.write(f'        def Xform "Group_{g}"\n        {{\n')
            for i in range(instances):
                k = (g * instances + i) % props
                meta = f"references = @./assets/prop_{k}.usda@"
                if instanceable:
                    meta += "\n                instanceable = true"
                lod = "low" if i % 3 == 0 else "high"
                f.write(f'            def "Inst_{i}" (\n                {meta}\n')
                f.write(f'                variants = {{\n                    string lod = "{lod}"\n                }}\n')
                f.write("            )\n            {\n")
                f.write(f"                double3 xformOp:translate = ({i}, {g}, 0)\n")
                f.write('                uniform token[] xformOpOrder = ["xformOp:translate"]\n')
                f.write("            }\n")
            f.write("        }\n")
        f.write("    }\n}\n")

    with open(os.path.join(out, "anim.usda"), "w") as f:
        f.write("#usda 1.0\n\n")
        f.write('over "World"\n{\n    over "Set"\n    {\n')
        for g in range(0, groups, 2):
            f.write(f'        over "Group_{g}"\n        {{\n')
            f.write("            double3 xformOp:translate.timeSamples = {\n")
            f.write("                1: (0, 0, 0),\n                24: (0, 0, 10),\n            }\n        }\n")
        f.write("    }\n}\n")

    with open(os.path.join(out, "shot.usda"), "w") as f:
        f.write('#usda 1.0\n(\n    defaultPrim = "World"\n    subLayers = [\n')
        f.write('        @./anim.usda@,\n        @./set.usda@\n    ]\n)\n\n')
        f.write('class "_class_Prop"\n{\n    custom int classOpinion = 1\n}\n')


if __name__ == "__main__":
    main()
