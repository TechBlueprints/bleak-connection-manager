# -*- coding: utf-8 -*-
"""Tests for the mgmt-socket connection-parameter slice."""

import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bleak_connection_manager import mgmt  # noqa: E402

ADDRESS = "C8:47:8C:00:00:01"


def test_the_load_conn_param_packet_matches_the_mgmt_wire_format():
    packet = mgmt._conn_param_packet(3, ADDRESS, mgmt.FAST_CONN_PARAMS)

    opcode, index, length = struct.unpack("<HHH", packet[:6])
    assert opcode == mgmt.MGMT_OP_LOAD_CONN_PARAM == 0x0035
    assert index == 3
    payload = packet[6:]
    assert length == len(payload)

    (count,) = struct.unpack("<H", payload[:2])
    assert count == 2  # one entry per LE address type
    entry_size = 6 + 1 + 2 * 4
    entries = payload[2:]
    assert len(entries) == count * entry_size

    bdaddr, addr_type, mn, mx, lat, timeout = struct.unpack("<6sBHHHH", entries[:entry_size])
    assert bdaddr == bytes.fromhex("0100008C47C8")  # reversed byte order
    assert addr_type == mgmt.BDADDR_LE_PUBLIC
    assert (mn, mx, lat, timeout) == (0x06, 0x06, 0, 1000)  # 7.5ms, 10s

    _, addr_type_2, _, _, _, timeout_2 = struct.unpack("<6sBHHHH", entries[entry_size:])
    assert addr_type_2 == mgmt.BDADDR_LE_RANDOM
    assert timeout_2 == 1000


def test_the_medium_parameters_relax_the_fast_ones():
    fast = mgmt.FAST_CONN_PARAMS
    medium = mgmt.MEDIUM_CONN_PARAMS
    assert medium[0] > fast[0]  # longer interval
    assert medium[3] < fast[3]  # tighter supervision once established


def test_load_degrades_to_false_when_the_channel_is_unavailable(monkeypatch):
    monkeypatch.setattr(mgmt, "_available", False)
    assert mgmt.load_fast("hci0", ADDRESS) is False
    assert mgmt.load_medium("hci0", ADDRESS) is False


def test_load_rejects_non_hci_adapters_before_touching_the_socket(monkeypatch):
    monkeypatch.setattr(mgmt, "_available", True)  # would be used if reached
    monkeypatch.setattr(mgmt, "_sock", None)
    assert mgmt.load_conn_params("bogus", ADDRESS, mgmt.FAST_CONN_PARAMS) is False


def test_the_probe_never_raises_on_this_platform(monkeypatch):
    monkeypatch.setattr(mgmt, "_available", None)
    monkeypatch.setattr(mgmt, "_sock", None)
    assert mgmt.available() in (True, False)  # macOS/CI: typically False
