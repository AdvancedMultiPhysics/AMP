import re
import csv

def main():
    
    narray = [16, 32, 64, 128] #, 256]

    for n in narray:
        #data = parse_file(f'../build/out/disc_err_test_2D/n{n}.txt')
        #write_csv(data, f'out/disc_err_test_2D/n{n}.csv')

        data = parse_file(f'../build/out/disc_err_test_2D_adaptive_dt/n{n}.txt')
        write_csv(data, f'out/disc_err_test_2D_adaptive_dt/n{n}.csv')

def parse_file(filepath):
    with open(filepath) as f:
        content = f.read()

    header_pattern = re.compile(
        r"PDE\s*\{[^}]*?\bdim\s*=\s*(\d+).*?\bmodel\s*=\s*\"([^\"]+)\".*?\}.*?"
        r"mesh\s*\{[^}]*?\bdim\s*=\s*(\d+).*?\bh\s*=\s*([\d.]+).*?\bn\s*=\s*(\d+)",
        flags=re.MULTILINE | re.DOTALL
    )
    header = header_pattern.search(content)
    pde_dim = int(header.group(1))
    model = header.group(2)
    mesh_dim = int(header.group(3))
    h = float(header.group(4))
    n = int(header.group(5))

    # Updated regex to also capture dt and time
    record_pattern = re.compile(r'''
        Starting\ timestep\s*(\d+),\s*dt\s*=\s*([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?),\s*time\s*=\s*([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?).*?
        SNES\ Iterations\s*:\s*.*?nonlinear\s*:\s*(\d+).*?linear\s*:\s*(\d+).*?
        Manufactured\s+discretization\s+error\s+norms\s*:
        .*?\|\|e\|\|\s*=\s*\(\s*([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*,\s*
                         ([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*,\s*
                         ([+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*\)
    ''', flags=re.MULTILINE | re.DOTALL | re.VERBOSE)

    records = []
    for m in record_pattern.finditer(content):
        rec = {
            'timestep': int(m.group(1)),
            'dt': float(m.group(2)),
            'time': float(m.group(3)),
            'nonlinear_iters': int(m.group(4)),
            'linear_iters': int(m.group(5)),
            'err1': float(m.group(6)),
            'err2': float(m.group(7)),
            'err3': float(m.group(8))
        }
        rec['errmax'] = rec.pop('err3')
        records.append(rec)

    return {
        'pde_dim': pde_dim,
        'model': model,
        'mesh_dim': mesh_dim,
        'h': h,
        'n': n,
        'records': records
    }


def write_csv(parsed, out_csv):
    fieldnames1 = ['pde_dim', 'model', 'mesh_dim', 'h', 'n']
    fieldnames2 = ['timestep', 'dt', 'time', 'nonlinear_iters', 'linear_iters', 'err1', 'err2', 'errmax']

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames1)
        writer.writeheader()
        writer.writerow({
            'pde_dim': parsed['pde_dim'],
            'model': parsed['model'],
            'mesh_dim': parsed['mesh_dim'],
            'h': parsed['h'],
            'n': parsed['n']
        })

        writer = csv.DictWriter(f, fieldnames=fieldnames2)
        writer.writeheader()
        for rec in parsed['records']:
            writer.writerow(rec)


if __name__ == '__main__':
    main()
