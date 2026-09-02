import numpy as np
import mne

'''
This code maps the BioSemi 128-channel names (A1-D32) to the nearest standard 
10-05 system electrode names (around 300 channels) which is more channels than the 128
the data in this project has. However a more specific overlay can filter out the electrodes
that we might not need to extract data from. 


'''

biosemi = mne.channels.make_standard_montage("biosemi128")
standard = mne.channels.make_standard_montage("standard_1005")

biosemi_pos = biosemi.get_positions()["ch_pos"]
standard_pos = standard.get_positions()["ch_pos"]

#sanity check: these values should be close to each other
biosemi_radius = np.mean([np.linalg.norm(p) for p in biosemi_pos.values()])
standard_radius = np.mean([np.linalg.norm(p) for p in standard_pos.values()])
print(f"Mean head radius -- biosemi128: {biosemi_radius:.4f} m, standard_1005: {standard_radius:.4f} m")


standard_names = list(standard_pos.keys())
standard_coords = np.array([standard_pos[ch] for ch in standard_names])

mapping = {}
for biosemi_ch, pos in biosemi_pos.items():
    dists = np.linalg.norm(standard_coords - np.array(pos), axis=1)
    nearest_idx = np.argmin(dists)
    mapping[biosemi_ch] = (standard_names[nearest_idx], dists[nearest_idx])

def biosemi_sort_key(ch):
    return (ch[0], int(ch[1:]))

print(f"{'BioSemi':<8} {'Nearest 10-05 name':<10} {'Distance (m)'}")
for ch in sorted(mapping, key=biosemi_sort_key):
    name, dist = mapping[ch]
    print(f"{ch:<8} {name:<10} {dist:.4f}")

#anatomical features to focus on
PREFIXES = ("FC", "FT", "T", "P", "TP", "CP")


def matches_prefix(name, prefixes):
    for p in sorted(prefixes, key=len, reverse=True):
        if name.startswith(p):
            return True
    return False

region_channels = sorted(
    [ch for ch, (name, _) in mapping.items() if matches_prefix(name, PREFIXES)],
    key=biosemi_sort_key,
)


print(f"\nRegion BioSemi channels ({len(region_channels)}): {region_channels}")

#flag any large distances
DIST_WARN_THRESHOLD = 0.015  # meters; adjust if everything looks fine but this is noisy
large_dist = [(ch, name, d) for ch, (name, d) in mapping.items() if d > DIST_WARN_THRESHOLD]
if large_dist:
    print(f"\n{len(large_dist)} channels matched with distance > {DIST_WARN_THRESHOLD} m, worth a closer look:")
    for ch, name, d in sorted(large_dist, key=lambda x: -x[2]):
        print(f"  {ch} -> {name} ({d:.4f} m)")
else:
    print(f"\nAll channels matched within {DIST_WARN_THRESHOLD} m, mapping looks solid.")
