"""Discriminating tests for archived trajectory identity and complete log parsing."""
import io
import os
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stats'))
import paper3_stats as D
from provenance import prepare_trajectories, trajectory_manifest, verify_trajectories


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def archive(self, duplicate=False):
        path = os.path.join(self.root, 'trajectories.tar.gz')
        with tarfile.open(path, 'w:gz') as t:
            for name, data in [('ep_00000.pkl', b'first'), ('ep_00001.pkl', b'second')]:
                member = tarfile.TarInfo('./' + name)
                member.size = len(data)
                t.addfile(member, io.BytesIO(data))
                if duplicate:
                    t.addfile(member, io.BytesIO(data))
        return path

    def test_supplied_copy_requires_identical_payload_set_and_bytes(self):
        archive = self.archive()
        directory, manifest = prepare_trajectories(archive, self.root)
        _, supplied_manifest = prepare_trajectories(archive, self.root, directory)
        self.assertEqual(manifest, supplied_manifest)
        with open(os.path.join(directory, 'ep_00000.pkl'), 'wb') as f:
            f.write(b'modified')
        with self.assertRaisesRegex(ValueError, 'changed='):
            prepare_trajectories(archive, self.root, directory)
        os.remove(os.path.join(directory, 'ep_00000.pkl'))
        with open(os.path.join(directory, 'ep_99999.pkl'), 'wb') as f:
            f.write(b'extra')
        with self.assertRaisesRegex(ValueError, 'missing=.*ep_00000.*extra=.*ep_99999'):
            verify_trajectories(directory, trajectory_manifest(archive))

    def test_ambiguous_archive_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            trajectory_manifest(self.archive(duplicate=True))

    def parse(self, contents):
        logs = os.path.join(self.root, 'results', 'logs')
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, 'test.log'), 'w') as f:
            f.write(contents)
        D.ROOT = self.root
        D.INPUTS.clear()
        return D.parse_eval('test.log', 'fixture')

    @staticmethod
    def log():
        return ('Episode 0 ends in 12\nSuccess\n\nEpisode 1 ends in 8\nCollision with human\n'
                '1 1 0\nCollision cases: 1\nCollision with Human cases: 1\n'
                'Collision with Obstacle cases:\nCollision with Wall cases:\nTimeout cases:\n'
                'Evaluation using 2 episodes\n')

    def test_parser_rejects_duplicate_and_incomplete_episode(self):
        ev = self.parse(self.log())
        self.assertEqual(ev['succ'], 1)
        self.assertTrue(ev['validation']['count_summary_checked'])
        with self.assertRaisesRegex(ValueError, 'duplicate episode'):
            self.parse('Episode 0 ends in 12\nSuccess\n' + self.log())
        with self.assertRaisesRegex(ValueError, 'no outcome'):
            self.parse(self.log().replace('Success\n', ''))

    def test_parser_crosschecks_count_summary_and_aggregate_list(self):
        with self.assertRaisesRegex(AssertionError, 'count summary'):
            self.parse(self.log().replace('1 1 0', '0 2 0'))
        with self.assertRaisesRegex(AssertionError, 'total collision list'):
            self.parse(self.log().replace('Collision cases: 1', 'Collision cases: 0'))


if __name__ == '__main__':
    unittest.main()
