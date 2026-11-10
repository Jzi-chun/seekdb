import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/dd/data.log'
data = open(path, 'rb').read()
print('file size', len(data))
magic, ver = struct.unpack_from('<IH', data, 0)
print('magic %08x version %d' % (magic, ver))
pos = 8
while pos + 25 <= len(data):
    blen, kind, tid, rowid, pk = struct.unpack_from('<IBIQQ', data, pos)
    body = data[pos+25:pos+25+blen]
    extra = ''
    if kind == 1 and len(body) > 8:
        # print varchar fields crudely
        extra = body[:8].hex()
    print('off=%d len=%d kind=%d tid=%d rowid=%d pk=%d bodylen=%d %s' % (pos, blen, kind, tid, rowid, pk, len(body), extra))
    pos += 25 + blen
print('stopped at', pos)
