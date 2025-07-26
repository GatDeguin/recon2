#!/usr/bin/env python3
import h5py, argparse

def inspect_h5(path):
    with h5py.File(path, 'r') as f:
        def recurse(name, obj, indent=0):
            prefix = '  ' * indent
            if isinstance(obj, h5py.Group):
                print(f"{prefix}Group: {name or '/'}")
                # atributos de grupo
                for k,v in obj.attrs.items():
                    print(f"{prefix}  • ATTR: {k} = {v!r}")
                for child in obj:
                    recurse(child, obj[child], indent+1)
            elif isinstance(obj, h5py.Dataset):
                print(f"{prefix}Dataset: {name}")
                print(f"{prefix}  shape: {obj.shape}")
                print(f"{prefix}  dtype: {obj.dtype}")
                if obj.chunks:
                    print(f"{prefix}  chunks: {obj.chunks}")
                if obj.compression:
                    print(f"{prefix}  compression: {obj.compression}")
                for k,v in obj.attrs.items():
                    print(f"{prefix}  • ATTR: {k} = {v!r}")
            else:
                print(f"{prefix}{name}: <Unknown HDF5 type>")

        recurse('', f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("h5file", help="ruta al .h5")
    args = p.parse_args()
    inspect_h5(args.h5file)
