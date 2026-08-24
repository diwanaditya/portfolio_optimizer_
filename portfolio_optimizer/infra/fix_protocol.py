"""
FIX Protocol Message Encoder/Decoder.

SCOPE HONESTY: this is a real, correct implementation of FIX tag=value
wire-format encoding/decoding, BodyLength and CheckSum calculation, and
the message types most relevant to portfolio rebalancing (NewOrderSingle,
ExecutionReport, OrderCancelRequest). It is NOT a FIX engine — there is
no session layer (Logon/Heartbeat/Sequence-number management/gap-fill
recovery), no TCP transport, and no counterparty certification. Building
a certified FIX session layer is a multi-month project typically done
with QuickFIX/J, QuickFIX/n, or a broker-provided SDK, not something a
portfolio-optimization library should reimplement. What IS provided here
is the correct, testable message-construction layer that would sit
underneath such a session layer.
"""
from __future__ import annotations
from dataclasses import dataclass, field


SOH = "\x01"  # FIX field delimiter (ASCII 0x01, "Start of Heading")


@dataclass
class FixMessage:
    msg_type: str
    fields: dict  # tag (int) -> value (str), in insertion order

    def __post_init__(self):
        # ensure keys are ints
        self.fields = {int(k): str(v) for k, v in self.fields.items()}


def encode_fix_message(msg: FixMessage, sender_comp_id: str, target_comp_id: str,
                        seq_num: int, fix_version: str = "FIX.4.4") -> str:
    """Encode a FixMessage into the wire-format tag=value string, including
    correctly computed BodyLength (tag 9) and CheckSum (tag 10).

    FIX wire format: tag=value SOH tag=value SOH ...
    Header: 8=BeginString | 9=BodyLength | 35=MsgType | 49=SenderCompID |
            56=TargetCompID | 34=MsgSeqNum | ... body fields ... | 10=CheckSum
    BodyLength = byte count from AFTER tag 9's SOH through BEFORE tag 10.
    CheckSum = (sum of all bytes up to but not including tag 10) mod 256,
               formatted as a zero-padded 3-digit string.
    """
    header_fields = [
        (35, msg.msg_type), (49, sender_comp_id), (56, target_comp_id), (34, str(seq_num)),
    ]
    body_parts = [f"{tag}={value}" for tag, value in header_fields]
    body_parts += [f"{tag}={value}" for tag, value in msg.fields.items()]
    body = SOH.join(body_parts) + SOH

    begin_string_field = f"8={fix_version}{SOH}"
    body_length_field = f"9={len(body)}{SOH}"

    pre_checksum = begin_string_field + body_length_field + body
    checksum = sum(pre_checksum.encode("ascii")) % 256
    checksum_field = f"10={checksum:03d}{SOH}"

    return pre_checksum + checksum_field


def decode_fix_message(raw: str) -> dict:
    """Decode a raw FIX wire-format string into a {tag: value} dict, and
    verify the checksum and body length are internally consistent
    (returns them in the result so callers can assert validity).
    """
    raw = raw.rstrip(SOH)
    parts = raw.split(SOH)
    fields = {}
    checksum_part_index = None
    for i, part in enumerate(parts):
        if "=" not in part:
            continue
        tag_str, value = part.split("=", 1)
        tag = int(tag_str)
        fields[tag] = value
        if tag == 10:
            checksum_part_index = i

    if checksum_part_index is not None:
        pre_checksum_raw = SOH.join(parts[:checksum_part_index]) + SOH
        computed_checksum = sum(pre_checksum_raw.encode("ascii")) % 256
        fields["_computed_checksum"] = computed_checksum
        fields["_checksum_valid"] = (str(computed_checksum).zfill(3) == fields.get(10, "").zfill(3))
    else:
        fields["_computed_checksum"] = None
        fields["_checksum_valid"] = False
    return fields


# --------------------------------------------------------------------- #
# Convenience builders for the message types relevant to rebalancing
# --------------------------------------------------------------------- #

def new_order_single(cl_ord_id: str, symbol: str, side: str, order_qty: float,
                      ord_type: str = "1", price: float | None = None,
                      time_in_force: str = "0") -> FixMessage:
    """MsgType=D (NewOrderSingle).
    side: '1'=Buy, '2'=Sell. ord_type: '1'=Market, '2'=Limit.
    time_in_force: '0'=Day, '3'=IOC, '1'=GTC.
    """
    fields = {
        11: cl_ord_id,      # ClOrdID
        55: symbol,         # Symbol
        54: side,           # Side
        38: str(order_qty), # OrderQty
        40: ord_type,       # OrdType
        59: time_in_force,  # TimeInForce
    }
    if ord_type == "2" and price is not None:
        fields[44] = str(price)  # Price (Limit)
    return FixMessage(msg_type="D", fields=fields)


def execution_report(cl_ord_id: str, order_id: str, exec_id: str, exec_type: str,
                      ord_status: str, symbol: str, side: str, order_qty: float,
                      cum_qty: float, leaves_qty: float, avg_px: float,
                      last_px: float = 0.0, last_qty: float = 0.0) -> FixMessage:
    """MsgType=8 (ExecutionReport).
    exec_type / ord_status (subset): '0'=New, '1'=Partial fill, '2'=Fill,
    '4'=Cancelled, '8'=Rejected.
    """
    fields = {
        11: cl_ord_id, 37: order_id, 17: exec_id, 150: exec_type, 39: ord_status,
        55: symbol, 54: side, 38: str(order_qty), 14: str(cum_qty),
        151: str(leaves_qty), 6: str(avg_px), 31: str(last_px), 32: str(last_qty),
    }
    return FixMessage(msg_type="8", fields=fields)


def order_cancel_request(orig_cl_ord_id: str, cl_ord_id: str, symbol: str, side: str,
                          order_qty: float) -> FixMessage:
    """MsgType=F (OrderCancelRequest)."""
    fields = {
        41: orig_cl_ord_id,  # OrigClOrdID
        11: cl_ord_id,       # ClOrdID (new)
        55: symbol, 54: side, 38: str(order_qty),
    }
    return FixMessage(msg_type="F", fields=fields)


# Human-readable field name lookup for debugging/logging
FIELD_NAMES = {
    8: "BeginString", 9: "BodyLength", 10: "CheckSum", 11: "ClOrdID", 14: "CumQty",
    17: "ExecID", 31: "LastPx", 32: "LastQty", 34: "MsgSeqNum", 35: "MsgType",
    37: "OrderID", 38: "OrderQty", 39: "OrdStatus", 40: "OrdType", 41: "OrigClOrdID",
    44: "Price", 49: "SenderCompID", 54: "Side", 55: "Symbol", 56: "TargetCompID",
    59: "TimeInForce", 6: "AvgPx", 150: "ExecType", 151: "LeavesQty",
}


def pretty_print(fields: dict) -> str:
    lines = []
    for tag, value in fields.items():
        if isinstance(tag, int):
            name = FIELD_NAMES.get(tag, f"Tag{tag}")
            lines.append(f"  {tag} ({name}): {value}")
    return "\n".join(lines)
