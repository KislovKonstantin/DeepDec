from deepdec import EventSystem, Event, SystemEvents

class MockObserver:
    def update(self, event):
        self.last_event = event

def test_event_system():
    system = EventSystem()
    observer = MockObserver()
    system.register(observer)

    test_event = Event(SystemEvents.AGENT_STARTED, {"name": "test"})
    system.post(test_event)

    assert observer.last_event.name == SystemEvents.AGENT_STARTED