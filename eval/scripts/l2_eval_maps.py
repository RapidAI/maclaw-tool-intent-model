import json, numpy as np, sys
src = open('scripts/l2_calib.py').read(); exec(src.split("print('== raw space, current")[0])
def mapped(a, b):  # prompt-space thresholds implied by the raw constants under s' = a s + b
    return (0.78 - b) / a, 0.10 / a, (0.70 - b) / a, 0.05 / a
qm = json.load(open('out/l2_map_quantile.json')); cal = json.load(open('out/l2_calib.json'))
C = [('identity (naive flip)', 1.0, 0.0), ('quantile map (SYN, unlabeled)', qm['scale'], qm['shift'])] + [(f'constrained {k}', v['scale'], v['shift']) for k, v in cal.items()]
print(f"{'map':32s} {'a':>6s} {'b':>7s} | t_p g_p tl_p gl_p | " + ' | '.join(f'{s}: cov/sel/n' for s in ('SYN', 'old', 'G', 'K')))
print(f"{'raw space, current constants':32s} {'':6s} {'':7s} | 0.780 0.100 0.700 0.050 | " + ' | '.join('%.3f/%.3f/%d' % stats(SV['raw', s], .78, .10, .70, .05) for s in ('SYN', 'old', 'G', 'K')))
for nm, a, b in C:
    t, g, tl, gl = mapped(a, b)
    print(f"{nm:32s} {a:6.3f} {b:7.3f} | {t:.3f} {g:.3f} {tl:.3f} {gl:.3f} | " + ' | '.join('%.3f/%.3f/%d' % stats(SV['cls', s], t, g, tl, gl) for s in ('SYN', 'old', 'G', 'K')))
