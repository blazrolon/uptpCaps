from datetime import datetime

def fromHere():
    return datetime.utcnow()


def toHere(dt=datetime.utcnow()):

    _dt=datetime.utcnow()
    print(_dt-dt)
