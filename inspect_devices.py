import soundcard as sc
import importlib

print('Inspecting sound devices...')

# soundcard default speaker
try:
    default = sc.default_speaker()
    print('Default speaker:')
    print('  name:', getattr(default, 'name', None))
    print('  id  :', getattr(default, 'id', None))
except Exception as e:
    print('Could not get default speaker:', e)

print('\nAll microphones (include_loopback=True):')
for mic in sc.all_microphones(include_loopback=True):
    print('---')
    print('name      :', getattr(mic, 'name', None))
    print('id        :', getattr(mic, 'id', None))
    print('isloopback:', getattr(mic, 'isloopback', None))
    print('channels  :', getattr(mic, 'channels', None))

# Try pycaw default render device (friendly name) as extra info
if importlib.util.find_spec('pycaw') is not None:
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities
        speakers = AudioUtilities.GetSpeakers()
        try:
            friendly = getattr(speakers, 'FriendlyName', None) or getattr(speakers, 'name', None)
        except Exception:
            friendly = None
        print('\npycaw default render device friendly name:', friendly)
    except OSError as ose:
        print('\npycaw/comtypes could not initialize due to COM threading (OSError):', ose)
    except Exception as e:
        print('\npycaw error:', e)
else:
    print('\npycaw not installed')
