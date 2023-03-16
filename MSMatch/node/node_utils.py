import pykep as pk
from paseos import ActorBuilder, SpacecraftActor

def parse_actor_data(self, actor_data):
    """Decode an actor from a data list

    Args:
        actor_data (list): [name,epoch,pos,velocity]

    Returns:
        actor: Created actor
    """
    actor = ActorBuilder.get_actor_scaffold(
        name=actor_data[0], actor_type=SpacecraftActor, epoch=actor_data[1]
    )
    earth = pk.planet.jpl_lp("earth")
    ActorBuilder.set_orbit(
        actor=actor,
        position=actor_data[2],
        velocity=actor_data[3],
        epoch=actor_data[1],
        central_body=earth,
    )
    return actor

def exchange_actors(node, verbose=False):
        """This function exchanges the states of various nodes among all MPI ranks.

        Args:
            comm (MPI_COMM_WORLD): The MPI comm world.
            paseos_instance (PASEOS): The local paseos instance.
            local_actor (SpacecraftActor): The rank's local actor.
            other_ranks (list of int): The indices of the other ranks.
            rank (int): Rank's index.
        """
        if verbose:
            print(f"Rank {node.rank} starting actor exchange.", flush=True)
        send_requests = []  # track our send requests
        recv_requests = []  # track our receive request
        node.paseos.emtpy_known_actors()  # forget about previously known actors

        # Send local actor to other ranks
        for i in node.other_ranks:
            actor_data = node._encode_actor()
            send_requests.append(
                node.comm.isend(actor_data, dest=i, tag=int(str(node.rank) + str(i)))
            )

        # Receive from other ranks
        for i in node.other_ranks:
            recv_requests.append(
                node.comm.irecv(source=i, tag=int(str(i) + str(node.rank)))
            )

        # Wait for data to arrive
        window_end = pk.epoch(
            node.local_actor.local_time.mjd2000 + node.comm_duration * pk.SEC2DAY
        )

        node.ranks_in_lineofsight = []
        window_start = node.local_actor.local_time
        window_end = pk.epoch(
            node.local_actor.local_time.mjd2000 + node.comm_duration * pk.SEC2DAY
        )
        for i, recv_request in enumerate(recv_requests):
            other_actor_data = recv_request.wait()
            other_actor = parse_actor_data(other_actor_data)

            if node.local_actor.is_in_line_of_sight(
                other_actor, epoch=window_start
            ) and node.local_actor.is_in_line_of_sight(other_actor, epoch=window_end):
                node.paseos.add_known_actor(other_actor)
                node.ranks_in_lineofsight.append(node.other_ranks[i])

        # Wait until all other ranks have received everything.
        for send_request in send_requests:
            send_request.wait()

        if verbose:
            print(
                f"Rank {node.rank} completed actor exchange. Knows {node.paseos.known_actor_names} now.",
                flush=True,
            )

# ToDo: put server on separate MPI process and let nodes announce to server that model has been updated
# ToDo: Also, remove the IO from server class
def announce_update(node):