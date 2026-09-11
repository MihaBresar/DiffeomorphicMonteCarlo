"""Build the manuscript's combined posterior RMSE table from saved summaries."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root', type=Path)
    args = parser.parse_args()
    rows, csv_rows = [], []
    for name, label in [('eight-schools', 'Centered Eight Schools'), ('garch11', 'GARCH(1,1)')]:
        payload = json.loads((args.output_root/name/'summary.json').read_text())
        algorithms = payload['scenarios'][0]['algorithms']
        vals = [algorithms[method][f'{moment}_moment_rmse_across_chains_and_coordinates'] for moment in ['first', 'second'] for method in ['NUTS', 'DCS-HnR']]
        rows.append(label+' & '+' & '.join(f'{v:.2e}' for v in vals)+r' \\')
        csv_rows.append([label]+vals)
    preamble = '% Generated from independent-chain errors relative to estimated PosteriorDB references.\n'
    table = '\\begin{tabular}{lrrrr}\nModel & NUTS first & DCS first & NUTS second & DCS second \\\\\n\\hline\n'+'\n'.join(rows)+'\n\\end{tabular}\n'
    path = args.output_root/'posteriordb_rmse.tex'
    path.write_text(preamble+table)
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'NUTS_first', 'DCS_first', 'NUTS_second', 'DCS_second'])
        writer.writerows(csv_rows)
    print(path.resolve())


if __name__ == '__main__':
    main()
