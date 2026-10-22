"""Read only the official RHD ZIP directory using HTTP byte ranges."""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.bytes_received = 0
        self.requests = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError("negative seek")
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = self.size-self.position
        if count == 0 or self.position >= self.size:
            return b""
        end = min(self.size, self.position+count)-1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with urllib.request.urlopen(req, timeout=45) as response:
            if response.status != 206:
                raise ValueError("server does not support byte ranges")
            value = response.read()
        if len(value) != end-self.position+1:
            raise ValueError("partial HTTP range")
        self.requests.append([self.position,end])
        self.position += len(value)
        self.bytes_received += len(value)
        return value


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    remote = HTTPRangeFile(URL,7080690299)
    with zipfile.ZipFile(remote) as archive:
        info = [{"name":i.filename,"bytes":i.file_size,"compressed":i.compress_size,
                 "offset":i.header_offset,"crc":i.CRC} for i in archive.infolist()]
    result = {"url":URL,"archive_size":remote.size,"directory_bytes_received":remote.bytes_received,
              "entries":info}
    (root/"artifacts/rhd_archive_index.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"entries":len(info),"directory_bytes_received":remote.bytes_received,
                      "metadata":[x for x in info if not x["name"].endswith(".png")][:30]},indent=2))
