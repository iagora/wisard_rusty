local zlib = require "zlib"
local struct = require "struct"

function load_mnist(folder, prefix)
    data_file = io.open(folder .. prefix .. '-images-idx3-ubyte.gz', "rb")
    io.input(data_file)
    data_buffer_gz = io.read("*all")
    inflater = zlib.inflate()
    data_buffer = inflater(data_buffer_gz)
    data = struct.unpack('>I4',data_buffer)
    metadata_bytes = 64

    label_file = io.open(folder .. prefix .. '-labels-idx1-ubyte.gz', "rb")
    io.input(label_file)
    label_buffer_gz = io.read("*all")
    inflater = zlib.inflate()
    label_buffer = inflater(label_buffer_gz)
    
end

wrk.method = "POST"
wrk.headers["Content-Type"] = "multipart/form-data; boundary=" .. Boundary
wrk.body = BodyBoundary .. CRLF .. ContentDisposition .. CRLF .. CRLF .. FileBody .. CRLF .. LastBoundary
