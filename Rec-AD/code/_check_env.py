import importlib
import sys

for name in ["numpy", "pandas", "sklearn", "torch", "yaml", "matplotlib", "pandapower", "pynvml"]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print("{:<12} OK      {}".format(name, ver))
    except Exception:
        print("{:<12} MISSING".format(name))

print()
print("python", sys.version.split()[0])
