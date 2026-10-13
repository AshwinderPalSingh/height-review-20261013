"""Verify trajectory payloads against their archived container before analysis."""
import glob
import hashlib
import os
import tarfile


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def trajectory_manifest(archive):
    """Hash all root ep_*.pkl members, rejecting ambiguous names or links."""
    result = {}
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            name = member.name.removeprefix('./')
            if not name.startswith('ep_') or not name.endswith('.pkl'):
                continue
            if '/' in name or not member.isfile() or name in result:
                raise ValueError('Ambiguous trajectory member: ' + member.name)
            with t.extractfile(member) as f:
                result[name] = hashlib.sha256(f.read()).hexdigest()
    if not result:
        raise ValueError('No trajectory payloads in ' + archive)
    return result


def verify_trajectories(directory, expected):
    """Require the exact archive payload set and bytes, including supplied copies."""
    actual = {os.path.basename(p): file_sha256(p)
              for p in sorted(glob.glob(os.path.join(directory, 'ep_*.pkl')))}
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    changed = sorted(p for p in actual.keys() & expected.keys() if actual[p] != expected[p])
    if missing or extra or changed:
        raise ValueError('Trajectory copy differs from hashed archive: missing=%s extra=%s changed=%s'
                         % (missing[:5], extra[:5], changed[:5]))
    h = hashlib.sha256()
    for name, digest in sorted(actual.items()):
        h.update(name.encode())
        h.update(bytes.fromhex(digest))
    return {'n_files': len(actual), 'traj_dir_sha256': h.hexdigest(), 'payload_sha256': actual}


def prepare_trajectories(archive, temp_dir, supplied=None):
    expected = trajectory_manifest(archive)
    directory = supplied or os.path.join(temp_dir, os.path.basename(archive).replace('.tar.gz', ''))
    if supplied is None:
        os.makedirs(directory, exist_ok=True)
        # Extract only the verified regular pickle payloads, without using path metadata.
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                name = member.name.removeprefix('./')
                if name in expected:
                    with t.extractfile(member) as src, open(os.path.join(directory, name), 'wb') as dst:
                        dst.write(src.read())
    return directory, verify_trajectories(directory, expected)
