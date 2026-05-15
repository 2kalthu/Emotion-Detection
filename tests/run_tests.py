import importlib, sys, os

# Ensure stdout uses UTF-8 to avoid UnicodeEncodeError on Windows consoles
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    mod = importlib.import_module('tests.test_love_detection')
except Exception as e:
    print('IMPORT_ERROR', e)
    raise

ok = True
for name in dir(mod):
    if name.startswith('test_'):
        fn = getattr(mod, name)
        try:
            fn()
            print(f'{name}: OK')
        except Exception as e:
            print(f'{name}: FAIL - {e}')
            ok = False

if not ok:
    sys.exit(1)

print('All tests passed')
