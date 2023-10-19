import json

import elasticache_auto_discovery
from pymemcache.client.hash import HashClient

from inference.core.env import ELASTICACHE_ENDPOINT
from inference.core.logger import logger

nodes = elasticache_auto_discovery.discover(ELASTICACHE_ENDPOINT)

# set up memcache
nodes = map(lambda x: (x[1], int(x[2])), nodes)
memcache_client = HashClient(nodes)


def trackUsage(endpoint, actor, n=1):
    """Tracks the usage of an endpoint by an actor.

    This function increments the usage count for a given endpoint by an actor.
    It also handles initialization if the count does not exist.

    Args:
        endpoint (str): The endpoint being accessed.
        actor (str): The actor accessing the endpoint.
        n (int, optional): The number of times the endpoint was accessed. Defaults to 1.

    Returns:
        None: This function does not return anything but updates the memcache client.
    """
    # count an inference
    try:
        job = endpoint + "endpoint:::actor" + actor
        current_infers = memcache_client.incr(job, n)
        if current_infers is None:  # not yet set; initialize at 1
            memcache_client.set(job, n)
            current_infers = n

            # store key
            job_keys = memcache_client.get("JOB_KEYS")
            if job_keys is None:
                memcache_client.add("JOB_KEYS", json.dumps([job]))
            else:
                decoded = json.loads(job_keys)
                decoded.append(job)
                decoded = list(set(decoded))
                memcache_client.set("JOB_KEYS", json.dumps(decoded))

            actor_keys = memcache_client.get("ACTOR_KEYS")
            if actor_keys is None:
                ak = {}
                ak[actor] = n
                memcache_client.add("ACTOR_KEYS", json.dumps(ak))
            else:
                decoded = json.loads(actor_keys)
                if actor in actor_keys:
                    actor_keys[actor] += n
                else:
                    actor_keys[actor] = n
                memcache_client.set("ACTOR_KEYS", json.dumps(actor_keys))

    except Exception as e:
        logger.debug("WARNING: there was an error in counting this inference")
        logger.debug(e)
