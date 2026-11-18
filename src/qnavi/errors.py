class QnaviError(Exception):
    pass


class ConfigError(QnaviError):
    pass


class RegistryError(QnaviError):
    pass


class CheckpointError(QnaviError):
    pass


class InvalidActionError(QnaviError):
    pass


class EnvironmentStateError(QnaviError):
    pass
