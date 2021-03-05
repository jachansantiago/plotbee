import bisect

class Track():
    
    def __init__(self, body):
        b = body
        self._start = body

        
        self._end = b
        self._event = None
        self._track_shape = None
        self.pollen = False
        self._tag = None
        self.tags_loaded = False

        # self.init()
        

    def init(self):
        self._tags = list()
        b = self._start

        if b.tag is not None:
            self._tags.append(b.tag)

        while b.next is not None:
            b = b.next
            if b.tag is not None:
                self._tags.append(b.tag)
            
        
        if len(self._tags) > 0:
            self._tag = self._tags[0]

        self.tags_loaded = True

    @property
    def tags(self):

        if not self.tags_loaded:

            self._tags = list()
            b = self._start

            while b.next is not None:
                b = b.next
                if b.tag is not None:
                    self._tags.append(b.tag)
            self.tags_loaded = True


        return self._tags

    @property
    def tag(self):
        if len(self.tags) > 0:
            self._tag = self.tags[0]
        return self._tag

    @property
    def id(self):
        return self._start.id

    @property
    def tag_id(self):
        if self.tag is None:
            return None
        else:
            return self.tag["id"]

    @property
    def end(self):
        if self._end.next is None:
            return self._end
        while self._end.next is not None:
            self._end = self._end.next
        return self._end

    def params(self):
        p = {
            "event": self._event,
            "track_shape": self._track_shape,
            "pollen": self.pollen
        }

        return p

    @property
    def start(self):
        return self._start

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, value):
        self._event = value

    @property
    def track_shape(self):
        return self._track_shape

    @track_shape.setter
    def track_shape(self, value):
        self._track_shape = value
    

    def __len__(self):
        self._size = 1
        x = self._start
        while x.next is not None:
            x = x.next
            self._size += 1
        return self._size
    
    def __getitem__(self, index):
        if index < self._size:
            x = self._start
            for _ in range(index):
                x = x.next
            return x
        else:
            raise IndexError("Index out of range.")

    def __iter__(self):
        x = self._start

        yield x

        while x.next is not None:
            x = x.next
            yield x
    
    def __repr__(self):
        return "Track({}, len={})".format(self.id, len(self))