# StratSim

add_executable(strat_sim
    StratSim.cpp
    )
target_link_libraries(strat_sim PUBLIC rt pthread floorlib algolib stratlib)

