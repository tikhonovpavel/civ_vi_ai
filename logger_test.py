from logger import Logger, MoveEvent, RangedAttackEvent

logger = Logger(map_size=(19, 19), map=[[[]]], initial_positions=[])

# -----------------------------------

logger.start_turn("Rome")
logger.log_event(MoveEvent("tank", (25, 13), path=[(25, 13), (25, 14)]))
logger.log_event(MoveEvent("tank", (10, 3), path=[(10, 3), (10, 4), (10, 5)]))

logger.start_turn("Egypt")
logger.log_event(RangedAttackEvent("artillery", (14, 4), (15, 4), enemy_damage=25))

# logger.end_game()

print(logger.to_json())
# with open('logs/logger_test.json', 'w') as f:
#     f.write(logger.to_json())