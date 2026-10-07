"""Re-materialize the anchor-only control heads reported in section 2 (same recipe/seed as train_heads.py; trained on anchors only, never on new data)."""
import sys; sys.path.insert(0, 'scripts')
import train_heads as T
for kind in ('logreg', 'mlp'):
    r, _ = T.run('gemma', kind, train_src=('anchor',), export=f'heads/head_{kind}_gemma_anchoronly.json')
    print(kind, 'old-test top1 %.3f tau %.2f T %.3f' % (r['top1'], r['tau'], r['T']))
