from user_app.inference.tcp_reassembly import TCPStream


def test_out_of_order_retransmission_and_overlap_are_not_duplicated():
    stream = TCPStream()
    stream.add(106, b"world")
    stream.add(100, b"hello ")
    stream.add(103, b"lo world")
    assert stream.contiguous_chunks() == [b"hello world"]


def test_capture_gap_is_not_silently_joined():
    stream = TCPStream()
    stream.add(10, b"abc")
    stream.add(20, b"xyz")
    assert stream.contiguous_chunks() == [b"abc", b"xyz"]
