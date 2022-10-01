from pathlib import Path
import shutil
import subprocess
import concurrent.futures
import tempfile
import mdtraj as mdt

faspr_bin = "/home/gzappavigna/software/FASPR/FASPR"
scwrl_bin = "/home/gzappavigna/software/scwrl/Scwrl4"


# def run_faspr(tempdir, pdbs):
#     pdbs = []

#     for pdb_cg in enumerate(pdbs):
#         pdb_all = pdb_cg.parent / (pdb_cg.stem + "_all.pdb")

#         subprocess.run([faspr_bin, "-i", str(pdb_cg), "-o", str(pdb_all)], check=True)
#         assert pdb_all.exists()

#         pdbs.append(pdb_all)

#     return pdbs


def call_faspr(pdb):
    pdb_all = pdb.parent / (pdb.stem + "_all.pdb")

    subprocess.run([faspr_bin, "-i", str(pdb), "-o", str(pdb_all)], check=True)
    assert pdb_all.exists()

    return pdb_all


def call_scwrl(pdb):
    pdb_all = pdb.parent / (pdb.stem + "_all.pdb")

    subprocess.run(
        [scwrl_bin, "-i", str(pdb), "-o", str(pdb_all)],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    assert pdb_all.exists()

    return pdb_all


def add_sidechains(pdbs, method="scwrl"):
    method_funcs = {"scwrl": call_scwrl, "faspr": call_faspr}

    try:
        call_method = method_funcs[method]
    except KeyError:
        raise Exception

    with (
        tempfile.TemporaryDirectory() as tempdir_,
        concurrent.futures.ProcessPoolExecutor() as executor,
    ):
        tempdir = Path(tempdir_)
        futures = []

        for i, pdb in enumerate(pdbs):
            if isinstance(pdb, mdt.Trajectory):
                mdt_obj = pdb
            elif (pdb := Path(pdb)).exists():
                mdt_obj = mdt.load(pdb)

            new_pdb = tempdir / f"frame_{i}.pdb"
            mdt_obj.save(new_pdb)
            assert new_pdb.exists()

            # pdb = Path(shutil.copy(pdb, tempdir))

            future = executor.submit(call_method, new_pdb)
            futures.append(future)

        pdbs_all = []
        for future in concurrent.futures.as_completed(futures):
            pdbs_all.append(mdt.load(future.result()))

        # dones, not_dones = concurrent.futures.wait(futures)
        # assert len(not_dones) == 0
        # pdbs_all = [mdt.load(future.result()) for future in dones]

    return pdbs_all
