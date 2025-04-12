# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2014-2025 William Edwards <shadowapex@gmail.com>, Benjamin Bean <superman2k5@gmail.com>
from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tuxemon.db import db
from tuxemon.encounter import Encounter, EncounterData
from tuxemon.session import local_session

if TYPE_CHECKING:
    from tuxemon.player import Player
    from tuxemon.states.world.worldstate import WorldState

logger = logging.getLogger(__name__)


class BackgroundThreadHandler:
    """
    Handles a single background thread that runs continuously.
    The thread logs the player's current location/route and has 
    placeholder functionality for database access.
    """

    _instance: Optional[BackgroundThreadHandler] = None
    _thread: Optional[threading.Thread] = None
    _running: bool = False
    _current_map: str = ""
    _last_map_monsters: Dict[str, List[str]] = {}

    def __new__(cls) -> BackgroundThreadHandler:
        """
        Creates a singleton instance of the BackgroundThreadHandler.
        """
        if cls._instance is None:
            cls._instance = super(BackgroundThreadHandler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """
        Initialize the background thread handler.
        """
        # Initialization only happens once because of the singleton pattern
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._running = False
            self._thread = None
            self._current_map = ""
            self._last_map_monsters = {}
            
    def _background_thread_function(self) -> None:
        """
        The function that runs continuously in the background thread.
        Logs the player's current location and has placeholder for DB operations.
        """
        logger.error("Background monitoring thread started")
        while self._running:
            try:
                if self._current_map:
                    logger.error(f"Player is currently at: {self._current_map}")
                    
                    # Extract the route name correctly (last part before .tmx)
                    # Example: "C:\Art-ML-pokemon\Tuxemon\mods\tuxemon\maps\spyder_route1.tmx" -> "spyder_route1"
                    map_name = self._current_map.split('\\')[-1].split('/')[-1].replace('.tmx', '')
                    
                    # Log monsters for this route
                    self._log_monsters_for_route(map_name)
                    
                    # TODO: Database functionality to be implemented
                    # - Insert entry into player_locations table with timestamp
                    # - Track time spent in each location
                    # - Record player movement patterns
                    # - Store encounter rates and battle results
                
            except Exception as e:
                logger.error(f"Error in background thread: {e}")
                
            # Sleep for a while before the next update
            time.sleep(5)  # Check every 5 seconds
            
        logger.error("Background monitoring thread stopped")
    
    def _log_monsters_for_route(self, route_name: str) -> None:
        """
        Fetches and logs all Pokémon (Tuxemon) names available on the current route.
        
        Parameters:
            route_name: The name of the current route/map.
        """
        # Check if we already logged monsters for this route
        if route_name in self._last_map_monsters:
            return
            
        try:
            # Define potential encounter slugs based on route name
            potential_slugs = [
                route_name,
                f"spyder_{route_name}",
                "default_encounter"
            ]
            
            monsters_found = False
            
            # Try each potential slug
            for slug in potential_slugs:
                try:
                    # Use direct db lookup instead of creating Encounter objects
                    results = db.lookup(slug, table="encounter")
                    logger.error(f"Results: {results}")
                    
                    # Get monsters directly from results
                    if hasattr(results, 'monsters') and results.monsters:
                        # Extract just the monster names from EncounterItemModel objects
                        monster_names = []
                        for encounter_item in results.monsters:
                            if hasattr(encounter_item, 'monster'):
                                monster_name = encounter_item.monster
                                if monster_name not in monster_names:
                                    monster_names.append(monster_name)
                        
                        if monster_names:
                            monster_str = ', '.join(monster_names)
                            logger.error(f"Monsters available on {route_name} (from {slug}): {monster_str}")
                            # Cache monster names for this route
                            self._last_map_monsters[route_name] = monster_names
                            monsters_found = True
                            break
                    
                except Exception as e:
                    # Try the next slug
                    logger.error(f"Error trying slug {slug}: {e}")
                    continue
            
            # If no encounters found through encounter system, try to guess based on common monsters
            if not monsters_found:
                logger.error(f"No encounter data found for {route_name}. Showing common monsters instead.")
                common_monsters = ["rockitten", "tweesher", "aardorn", "pairagrin", "cataspike", "hydrone"]
                self._last_map_monsters[route_name] = common_monsters
                logger.error(f"Common Tuxemon you might encounter: {', '.join(common_monsters)}")
                
        except Exception as e:
            logger.error(f"Error getting monsters for route {route_name}: {e}")
    
    def update(self, world_state: Any) -> None:
        """
        Updates the background thread with the player's current location.
        
        Parameters:
            world_state: The current world state.
        """
        try:
            # Get the current map name
            if hasattr(world_state, 'current_map') and hasattr(world_state.current_map, 'filename'):
                new_map = world_state.current_map.filename
                
                # Only log when the map changes
                if new_map != self._current_map:
                    self._current_map = new_map
                    logger.error(f"Player moved to: {self._current_map}")
                    
                    # TODO: Database functionality - log map change event
                    # - Record time of map transition
                    # - Update player's current location
                    
        except Exception as e:
            logger.error(f"Error updating background thread: {e}")
    
    def start_thread(self) -> None:
        """
        Starts the background thread if it's not already running.
        Should be called once when the game starts.
        """
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._background_thread_function)
            self._thread.daemon = True  # Thread will exit when main program exits
            self._thread.start()
            logger.error("Started background monitoring thread")
    
    def stop_thread(self) -> None:
        """
        Stops the background thread if it's running.
        """
        if self._running:
            self._running = False
            logger.error("Stopping background monitoring thread")
            # No need to join since it's a daemon thread 