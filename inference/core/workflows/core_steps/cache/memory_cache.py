class WorkflowMemoryCache:
    cache = {}

    @classmethod
    def get_dict(cls, namespace):
        if namespace not in WorkflowMemoryCache.cache:
            WorkflowMemoryCache.cache[namespace] = {}

        return WorkflowMemoryCache.cache[namespace]

    @classmethod
    def clear_namespace(cls, namespace):
        if namespace in WorkflowMemoryCache.cache:
            del WorkflowMemoryCache.cache[namespace]
