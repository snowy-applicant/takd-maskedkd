"""Fallback uv download for hosts without curl/wget (standard library only, SHA-256 verified).

Usage: python3 scripts/fetch_uv.py <version> <destination-dir>
"""
import hashlib
import io
import os
import sys
import tarfile
import urllib.request


def main():
    version, destination = sys.argv[1], sys.argv[2]
    url = f"https://github.com/astral-sh/uv/releases/download/{version}/uv-x86_64-unknown-linux-gnu.tar.gz"
    data = urllib.request.urlopen(url, timeout=300).read()
    expected = urllib.request.urlopen(url + ".sha256", timeout=60).read().split()[0].decode()
    if hashlib.sha256(data).hexdigest() != expected:
        sys.exit("uv tarball SHA-256 mismatch")
    os.makedirs(destination, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            name = os.path.basename(member.name)
            if member.isfile() and name in ("uv", "uvx"):
                path = os.path.join(destination, name)
                with open(path, "wb") as stream:
                    stream.write(archive.extractfile(member).read())
                os.chmod(path, 0o755)


if __name__ == "__main__":
    main()
