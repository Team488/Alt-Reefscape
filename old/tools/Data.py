from typing import Any, List, Optional, TypeVar, Generic, Iterator, Iterable

T = TypeVar('T')

class CircularBuffer(Generic[T]):
    """
    A fixed-size circular buffer that automatically overwrites
    the oldest items when full
    """
    
    def __init__(self, bufferSize: int) -> None:
        """
        Initialize the circular buffer
        
        Args:
            bufferSize: The maximum number of items the buffer can hold
        """
        if bufferSize <= 0:
            raise ValueError("Buffer size must be positive")
            
        self.size: int = bufferSize
        self.head: int = 0  # Index where the next item will be read from
        self.tail: int = 0  # Index where the next item will be written to
        self.count: int = 0  # Number of items currently in the buffer
        self.backing: List[Optional[T]] = [None] * bufferSize

    def put(self, value: T) -> None:
        """
        Add a new item to the buffer
        
        Args:
            value: The item to add to the buffer
        
        Notes:
            If the buffer is full, the oldest item will be overwritten
        """
        self.backing[self.tail] = value
        
        # Move the tail pointer to the next position
        self.tail = (self.tail + 1) % self.size
        
        # If the buffer is full, move the head pointer as well
        if self.count == self.size:
            self.head = (self.head + 1) % self.size
        else:
            # Otherwise, increment the count
            self.count += 1
            
    def get(self) -> Optional[T]:
        """
        Remove and return the oldest item from the buffer
        
        Returns:
            The oldest item in the buffer, or None if the buffer is empty
        """
        if self.count == 0:
            return None
            
        value = self.backing[self.head]
        self.backing[self.head] = None
        
        # Move the head pointer to the next position
        self.head = (self.head + 1) % self.size
        
        # Decrement the count
        self.count -= 1
        
        return value
        
    def peek(self) -> Optional[T]:
        """
        Return the oldest item from the buffer without removing it
        
        Returns:
            The oldest item in the buffer, or None if the buffer is empty
        """
        if self.count == 0:
            return None
            
        return self.backing[self.head]
        
    def is_empty(self) -> bool:
        """
        Check if the buffer is empty
        
        Returns:
            True if the buffer is empty, False otherwise
        """
        return self.count == 0
        
    def is_full(self) -> bool:
        """
        Check if the buffer is full
        
        Returns:
            True if the buffer is full, False otherwise
        """
        return self.count == self.size
        
    def clear(self) -> None:
        """
        Remove all items from the buffer
        """
        self.backing = [None] * self.size
        self.head = 0
        self.tail = 0
        self.count = 0
        
    def __len__(self) -> int:
        """
        Get the number of items currently in the buffer
        
        Returns:
            The number of items in the buffer
        """
        return self.count
        
    def __iter__(self) -> Iterator[T]:
        """
        Iterate over the items in the buffer from oldest to newest
        
        Returns:
            An iterator over the items in the buffer
        """
        idx = self.head
        for _ in range(self.count):
            yield self.backing[idx]  # type: ignore
            idx = (idx + 1) % self.size
