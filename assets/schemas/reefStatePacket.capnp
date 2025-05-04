@0xb98c80dc27ce4d38;

struct ReefPacket {
    message @0 :Text;                     # String message
    timestamp @1 :Float64;                 # Timestamp for the frame
    observationsReef @2 :List(ReefUpdate);                  # List of observations of reef
    observationsAlgae @3 :List(AlgaeUpdate);                  # List of observations of blocking algae
}

struct ReefUpdate {
    apriltagid @0 :Int8;                          # What april tag this observation is based off of
    branchindex @1 :Int8;                  # Branch id of observation
    openconfidence @2 :Float64;                      # how confident the observation is that the slot is open
}

struct AlgaeUpdate {
    apriltagid @0 :Int8;                          # What april tag this observation is based off of
    occupiedconfidence @1 :Float64;                      # how confident the observation is that algae occipies there
}
