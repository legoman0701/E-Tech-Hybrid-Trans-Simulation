#!/usr/bin/env python3
import asyncio, json, secrets, string, uuid

HOST = "0.0.0.0"
PORT = 8765
ROOM_CODE_LEN = 6
ALPHABET = string.ascii_uppercase + string.digits

# --- State ---
rooms = {}   # code -> set of client ids
clients = {} # client_id -> {"writer", "reader", "room", "name"}

def gen_code():
    while True:
        code = "".join(secrets.choice(ALPHABET) for _ in range(ROOM_CODE_LEN))
        if code not in rooms:
            return code

async def send(writer, msg: dict):
    data = (json.dumps(msg) + "\n").encode("utf-8")
    writer.write(data)
    await writer.drain()

async def broadcast(room_code, msg: dict, exclude_id=None):
    if room_code not in rooms: return
    dead = []
    for cid in list(rooms[room_code]):
        if exclude_id is not None and cid == exclude_id:
            continue
        info = clients.get(cid)
        if not info: 
            dead.append(cid)
            continue
        try:
            await send(info["writer"], msg)
        except Exception:
            dead.append(cid)
    for cid in dead:
        await remove_client(cid, reason="disconnect")

async def remove_client(client_id, reason="leave"):
    info = clients.get(client_id)
    if not info:
        return
    room = info.get("room")
    writer = info.get("writer")
    name = info.get("name", "")
    try:
        if not writer.is_closing():
            await send(writer, {"type":"bye","reason":reason})
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
    except Exception:
        pass
    clients.pop(client_id, None)
    if room and room in rooms:
        rooms[room].discard(client_id)
        # informer les autres
        await broadcast(room, {"type":"player_left","id":client_id,"name":name})
        if not rooms[room]:
            # salle vide -> supprimer
            rooms.pop(room, None)

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = {"reader": reader, "writer": writer, "room": None, "name": f"Player{client_id}"}
    await send(writer, {"type":"welcome","id":client_id})
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                await send(writer, {"type":"error","message":"invalid_json"})
                continue

            mtype = msg.get("type")

            if mtype == "hello":
                name = str(msg.get("name","")).strip() or f"Player{client_id}"
                clients[client_id]["name"] = name
                await send(writer, {"type":"hello_ok","name":name})

            elif mtype == "create_room":
                if clients[client_id]["room"]:
                    await send(writer, {"type":"error","message":"already_in_room"})
                    continue
                code = gen_code()
                rooms[code] = set()
                rooms[code].add(client_id)
                clients[client_id]["room"] = code
                await send(writer, {"type":"room_created","code":code})
                # L'host est aussi un joueur présent
                await send(writer, {"type":"joined","code":code,"players":[]})

            elif mtype == "join_room":
                code = msg.get("code","").upper()
                if not code or code not in rooms:
                    await send(writer, {"type":"error","message":"room_not_found"})
                    continue
                if clients[client_id]["room"]:
                    await send(writer, {"type":"error","message":"already_in_room"})
                    continue
                # existing players list
                players = []
                for cid in rooms[code]:
                    players.append({"id":cid, "name":clients[cid]["name"]})
                rooms[code].add(client_id)
                clients[client_id]["room"] = code
                # informer les autres
                await broadcast(code, {"type":"player_joined","id":client_id,"name":clients[client_id]["name"]}, exclude_id=client_id)
                await send(writer, {"type":"joined","code":code,"players":players})

            elif mtype == "state":
                # relay position/state to others in room
                room = clients[client_id]["room"]
                if not room:
                    continue
                pos = msg.get("pos")
                await broadcast(room, {"type":"state","id":client_id,"pos":pos}, exclude_id=client_id)

            elif mtype == "leave":
                await remove_client(client_id, reason="leave_request")
                return

            else:
                await send(writer, {"type":"error","message":f"unknown_type:{mtype}"})
    except Exception:
        pass
    finally:
        await remove_client(client_id, reason="disconnect")

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    addr = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"Relay server listening on {addr}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped.")
