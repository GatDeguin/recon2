import grpc
from . import transcriber_pb2 as transcriber__pb2

class TranscriberStub(object):
    def __init__(self, channel):
        self.Transcribe = channel.unary_unary(
            '/transcriber.Transcriber/Transcribe',
            request_serializer=transcriber__pb2.VideoRequest.SerializeToString,
            response_deserializer=transcriber__pb2.TranscriptReply.FromString,
        )

class TranscriberServicer(object):
    async def Transcribe(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_TranscriberServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Transcribe': grpc.unary_unary_rpc_method_handler(
            servicer.Transcribe,
            request_deserializer=transcriber__pb2.VideoRequest.FromString,
            response_serializer=transcriber__pb2.TranscriptReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'transcriber.Transcriber', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


