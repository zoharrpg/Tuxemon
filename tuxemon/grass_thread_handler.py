# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2014-2025 William Edwards <shadowapex@gmail.com>, Benjamin Bean <superman2k5@gmail.com>
from __future__ import annotations

from datetime import datetime
import logging
import os
import shutil
import threading
import cv2
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tuxemon.db import db
from tuxemon.encounter import Encounter, EncounterData
from tuxemon.session import local_session

if TYPE_CHECKING:
    from tuxemon.player import Player
    from tuxemon.states.world.worldstate import WorldState

logger = logging.getLogger(__name__)


def remove_background_kmeans_and_floodfill(
        input_path: str,
        output_path: str,
        k_clusters: int = 3,
        floodfill: bool = True
) -> None:
    import cv2
    import numpy as np
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        raise ValueError(f"Cannot read image from: {input_path}")

    Z = image_bgr.reshape((-1, 3))
    Z = np.float32(Z)

    # K-Means parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        Z,
        k_clusters,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    labels_2d = labels.reshape((image_bgr.shape[0], image_bgr.shape[1]))

    height, width = labels_2d.shape
    edge_labels = np.concatenate([
        labels_2d[0, :],
        labels_2d[-1, :],
        labels_2d[:, 0],
        labels_2d[:, -1]
    ])
    cluster_counts = np.bincount(edge_labels)
    background_cluster = cluster_counts.argmax()

    mask = np.uint8(labels_2d != background_cluster)

    if floodfill:
        mask_ff = mask.copy()
        h, w = mask_ff.shape

        mask_ff_3ch = cv2.merge([mask_ff, mask_ff, mask_ff])

        # Provide a mask buffer for floodFill that is 2 px bigger
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        for y, x in corners:
            if mask_ff[y, x] == 0:
                cv2.floodFill(
                    mask_ff_3ch,
                    flood_mask,
                    seedPoint=(x, y),
                    newVal=(0, 0, 0),
                    loDiff=(5, 5, 5),
                    upDiff=(5, 5, 5),
                    flags=4 | cv2.FLOODFILL_MASK_ONLY
                )
        # After flood filling, convert back to single-channel (just pick one)
        mask_refined = mask_ff_3ch[:, :, 0]

        mask = (mask_refined > 0).astype(np.uint8)


    b, g, r = cv2.split(image_bgr)
    alpha = (mask * 255).astype(np.uint8)
    rgba = cv2.merge([b, g, r, alpha])


    cv2.imwrite(output_path, rgba)

    print(f"Saved background-removed RGBA to: {output_path}")

def remove_background_kmeans_connected_bg(
    input_path: str,
    output_path: str,
    k_clusters: int = 3
):
    import cv2
    import numpy as np

    
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        raise ValueError(f"Cannot read image from: {input_path}")


    Z = image_bgr.reshape((-1, 3))
    Z = np.float32(Z)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels_2d = labels.reshape((image_bgr.shape[0], image_bgr.shape[1]))
    h, w = labels_2d.shape

    top_row = labels_2d[0, :]
    bottom_row = labels_2d[h-1, :]
    left_col = labels_2d[:, 0]
    right_col = labels_2d[:, w-1]
    edge_labels = np.concatenate([top_row, bottom_row, left_col, right_col])
    cluster_counts = np.bincount(edge_labels)
    background_cluster_id = cluster_counts.argmax()

    cluster_mask = np.uint8(labels_2d == background_cluster_id)

    num_labels, labeled_cc = cv2.connectedComponents(cluster_mask, connectivity=8)
    
    edge_components = set()

    edge_components.update(labeled_cc[0, :].tolist())

    edge_components.update(labeled_cc[h-1, :].tolist())

    edge_components.update(labeled_cc[:, 0].tolist())

    edge_components.update(labeled_cc[:, w-1].tolist())
    
    final_bg_mask = np.isin(labeled_cc, list(edge_components)).astype(np.uint8)

    foreground_mask = 1 - final_bg_mask

    b, g, r = cv2.split(image_bgr)
    alpha = (foreground_mask * 255).astype(np.uint8)
    rgba = cv2.merge([b, g, r, alpha])


    cv2.imwrite(output_path, rgba)
    print(f"Saved background-removed RGBA to: {output_path}")




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
            self._created_monster = False
            
            self.update_monster_description("ai_generated_1", "???")
            
            
            # for the purpose of testing
            self.clean_encounter_file("spyder_route1", "pairagrin")
            # load encounter table
            db.reload("encounter")
            logger.error("*************** Init thread: Reloaded encounter table ***************")
            # log current relative path when running
            logger.error(f"Current relative path: {os.path.relpath(__file__)}")
            # path to project root:
            logger.error(f"Project root: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
            # check if root/tuxemon-art-ml-resource dir exists
            self.resource_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "tuxemon-art-ml-resource")
            if not os.path.exists(self.resource_dir):
                logger.error(f"Resource directory does not exist: {self.resource_dir}")
                # create it
                os.makedirs(self.resource_dir)
            else:
                logger.error(f"Resource directory exists: {self.resource_dir}")
            
    def _background_thread_function(self) -> None:
        """
        The function that runs continuously in the background thread.
        Logs the player's current location and has placeholder for DB operations.
        """
        if(self._created_monster):
            return
        logger.error("Background monitoring thread started")
        while self._running:
            try:
                if self._current_map:
                    logger.error(f"Player is currently at: {self._current_map}")
                    
                    # Extract the route name correctly (last part before .tmx)
                    # Example: "C:\Art-ML-pokemon\Tuxemon\mods\tuxemon\maps\spyder_route1.tmx" -> "spyder_route1"
                    map_name = self._current_map.split('\\')[-1].split('/')[-1].replace('.tmx', '')
                    
                    self._log_monsters_for_route(map_name)
                    
                
            except Exception as e:
                logger.error(f"Error in background thread: {e}")
                
            
            time.sleep(5)  # Check every 5 seconds
            
        logger.error("Background monitoring thread stopped")
    
    def _log_monsters_for_route(self, route_name: str) -> None:
        """
        Fetches and logs all Pokémon (Tuxemon) names available on the current route.
        
        Parameters:
            route_name: The name of the current route/map.
        """
        
        
        if self._created_monster:
            return
        self._created_monster = True
        
        from openai import OpenAI
        import base64

        # TODO: change to your own api key
        client = OpenAI(
            api_key = ""
        )
        # Read and encode your local reference image, the image is under resource_dir
        image_path = os.path.join(self.resource_dir, "sample.png")
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        context = "no other special context requirement, original is fine"
        context = "showcasing a different culture's folktales, traditional attire, or iconic landmarks." 
        logger.error(f"********** Context: {context} **********")
        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a prompt generator for AI image creation."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "Give me an original Pokémon style prompt like this example picture, the prompt must mention that the background is purely white, the whole imageshould only contain the front image of this single monster. "
                                ", no any other extra element. "
                                "Generate a new DALL-E prompt for an original but simple pixel-art monster inspired by this style"
                                f"Additional context requirement of the generated pokemon if any: {context}"
                            )
                        }
                    ]
                }
            ]
        )

        prompt = chat_response.choices[0].message.content
        logger.error(f"Generated prompt: {prompt}")
        
        
        chat_response2 = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative pokemon description generator."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "Previously I asked you to generate a prompt for a pokemon's image, now I want you to generate a description for this pokemon-like monster. "
                                 f"Your previous prompt for this pokemon: {prompt}\n\n"
                                 "The description should be a short and concise description of the pokemon-like monster, it should be 100 words or less. "
                                 f"Additional context requirement of the generated pokemon-like monster if any: {context}"
                                 "If there exists special context requirement, please consider how to make sense of it."
                                 
                            )
                        }
                    ]
                }
            ]
        )
        description = chat_response2.choices[0].message.content
        logger.error(f"Generated pokemondescription: {description}")

        # Step 2: Use the generated prompt to get an image
        image_response = client.images.generate(
            model="dall-e-3",  # or "dall-e-2" if you want the older model
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        image_url = image_response.data[0].url
        logger.error(f"Image URL: {image_url}")
        
        # create a folder under resource_dir, and name related to current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"generated_monster_{timestamp}"
        folder_path = os.path.join(self.resource_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        import requests
        # download path is under folder_path
        download_path = os.path.join(folder_path, "dalle3_output.png")
        response = requests.get(image_url)
        with open(download_path, "wb") as f:
            f.write(response.content)
        hard_coded_path = "C:\\Art-ML-pokemon\\Tuxemon\\mods\\tuxemon\\gfx\\sprites\\battle"
        hard_coded_path += f"\\ai_generated_1-front.png"
        from rembg import remove
        from PIL import Image
        # make a temp folder for the images
        input_path = download_path
        output_path = os.path.join(folder_path, "dalle3_transparent.png")
        output_path_64_64 = os.path.join(folder_path, "dalle3_tran_64_64.png")
        output_non_tran_64_64 = os.path.join(folder_path, "dalle3_non_tran_64_64.png")
        output_cv = os.path.join(folder_path, "dalle3_transparent_cv.png")
        output_cv_64_64 = os.path.join(folder_path, "dalle3_tran_64_64.png")
        
        
        with Image.open(input_path) as img:
            img2 = img.resize((64, 64), Image.NEAREST)
            img2.save(output_non_tran_64_64)
            img = img.convert("RGBA")
            output = remove(img)
            output.save(output_path)
            output = output.resize((64, 64), Image.NEAREST)
            output.save(output_path_64_64)
            remove_background_kmeans_connected_bg(input_path, output_cv, 10)
            with Image.open(output_cv) as img:
                img = img.resize((64, 64), Image.NEAREST)
                img.save(output_cv_64_64)
            # copy the output_cv to hard_coded_path
            shutil.copy(output_cv_64_64, hard_coded_path)
        logger.error(f"Refined generated image saved to: {hard_coded_path}")
        
        # add this monster so we can encounter it
        self.add_debug_monster_to_route("ai_generated_1", 100.0)
        # add monster description
        self.update_monster_description("ai_generated_1", description)
        self._created_monster = True
            
        try:
            
            slug = route_name
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
                        db.reload("monster")
                        db.reload("encounter")
                        logger.error(f"Reloaded monster and encounter tables")
                        # Cache monster names for this route
                        self._last_map_monsters[route_name] = monster_names
                        monsters_found = True
                
            except Exception as e:
                # Try the next slug
                logger.error(f"Error trying slug {slug}: {e}")
            
            # If no encounters found through encounter system, try to guess based on common monsters
            if not monsters_found:
                logger.error(f"Exception: No encounters found for {route_name}")
                
        except Exception as e:
            logger.error(f"Error getting monsters for route {route_name}: {e}")
    
    def add_debug_monster_to_route(self, monster_name: str, encounter_rate: float = 100.0) -> bool:
        import json
        import os
        
        if not self._current_map:
            logger.error("No current map set. Player must be in a map first.")
            return False
            
        # Extract route name from the map filename
        route_name = self._current_map.split('\\')[-1].split('/')[-1].replace('.tmx', '')
        
        # Check if monster exists in database
        try:
            db.lookup(monster_name, table="monster")
        except Exception as e:
            logger.error(f"Monster '{monster_name}' not found in database: {e}")
            return False
            
        # Find the encounter file
        encounter_file_path = os.path.join("mods", "tuxemon", "db", "encounter", f"{route_name}.json")
        
        if not os.path.exists(encounter_file_path):
            logger.error(f"Encounter file not found: {encounter_file_path}")
            return False
            
        try:
            # Read the current encounter file
            with open(encounter_file_path, 'r') as f:
                encounter_data = json.load(f)
                
            # Create a new monster entry with high encounter rate
            new_monster_entry = {
                "monster": monster_name,
                "encounter_rate": encounter_rate,
                "variables": [
                    {
                        "daytime": "true"
                    }
                ],
                "exp_req_mod": 1,
                "level_range": [
                    2,
                    4
                ]
            }
            new_monster_entry2 = {
                "monster": monster_name,
                "encounter_rate": encounter_rate,
                "variables": [
                    {
                        "daytime": "false"
                    }
                ],
                "exp_req_mod": 1,
                "level_range": [
                    2,
                    4
                ]
            }
            
            monster_exists = False
            for monster in encounter_data["monsters"]:
                if monster["monster"] == monster_name and monster["encounter_rate"] == encounter_rate:
                    monster_exists = True
                    break
                    
            if not monster_exists:
                # Add the new monster entry to the encounter data
                encounter_data["monsters"].append(new_monster_entry)
                encounter_data["monsters"].append(new_monster_entry2)
                
                # Write the updated encounter data back to the file
                with open(encounter_file_path, 'w') as f:
                    json.dump(encounter_data, f, indent=2)
                    
                logger.error(f"Added monster '{monster_name}' to route '{route_name}' with encounter rate {encounter_rate}")
                
                # Reload the encounter table to apply changes
                db.reload("encounter")
                
                return True
            else:
                logger.error(f"Monster '{monster_name}' already exists in route '{route_name}' with encounter rate {encounter_rate}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding monster to route: {e}")
            return False
            
    def create_new_monster(self, monster_name: str) -> bool:
        import json
        import os
        import shutil
        
        # Define paths
        monster_dir = os.path.join("mods", "tuxemon", "db", "monster")
        template_path = os.path.join(monster_dir, "aardorn.json")
        new_monster_path = os.path.join(monster_dir, f"{monster_name}.json")
        
        # Check if the template file exists
        if not os.path.exists(template_path):
            logger.error(f"Template monster file not found: {template_path}")
            return False
            
        # Check if the new monster file already exists
        if os.path.exists(new_monster_path):
            logger.error(f"Monster file already exists: {new_monster_path}")
            return False
            
        try:
            with open(template_path, 'r') as f:
                monster_data = json.load(f)
                
            monster_data["slug"] = monster_name
            monster_data["evolutions"] = []  # Empty evolutions list
            
            # write the new monster file
            with open(new_monster_path, 'w') as f:
                json.dump(monster_data, f, indent=4)
                
            logger.error(f"Created new monster '{monster_name}' based on aardorn template")
            
            # reload the monster table to apply changes
            db.reload("monster")
            
            return True
                
        except Exception as e:
            logger.error(f"Error creating new monster: {e}")
            return False
            
    def create_new_monster_with_description(self, monster_name: str, description: str = "") -> bool:
        import json
        import os
        
        # first create the monster
        if not self.create_new_monster(monster_name):
            return False
            
        if description:
            try:
                trans_dir = os.path.join("mods", "tuxemon", "db", "locale", "en_US")
                monster_trans_path = os.path.join(trans_dir, "monster.json")
                
                monster_id = f"{monster_name.lower()}_description"
                
                if os.path.exists(monster_trans_path):
                    with open(monster_trans_path, 'r') as f:
                        trans_data = json.load(f)
                        
                    trans_data[monster_id] = description
                    
                    with open(monster_trans_path, 'w') as f:
                        json.dump(trans_data, f, indent=4, sort_keys=True)
                        
                    logger.error(f"Added description for '{monster_name}' to translation file")
                else:
                    logger.error(f"Translation file not found: {monster_trans_path}")
                
            except Exception as e:
                logger.error(f"Error adding description for '{monster_name}': {e}")
                
        return True
    
    def update_monster_description(self, monster_name: str, description: str) -> bool:
        import json
        import os
        from tuxemon.locale import T
        
        try:
            trans_dir = os.path.join("mods", "tuxemon", "db", "locale", "en_US")
            monster_trans_path = os.path.join(trans_dir, "monster.json")
            
            monster_id = f"{monster_name.lower()}_description"
            
            if os.path.exists(monster_trans_path):
                with open(monster_trans_path, 'r') as f:
                    trans_data = json.load(f)
                    
                trans_data[monster_id] = description
                
                with open(monster_trans_path, 'w') as f:
                    json.dump(trans_data, f, indent=4, sort_keys=True)
                
                if hasattr(T, 'reload'):
                    T.reload()
                    logger.error(f"Reloaded translations after updating '{monster_name}' description")
                elif hasattr(T, 'languages') and hasattr(T, 'build_translations'):
                    T.languages = {}
                    T.build_translations()
                    logger.error(f"Rebuilt translations after updating '{monster_name}' description")
                elif hasattr(T, 'languages'):
                    for lang in getattr(T, 'languages', {}).values():
                        lang[monster_id] = description
                    logger.error(f"Updated in-memory translation for '{monster_name}' description")
                else:
                    logger.error("Couldn't find a way to reload translations. Game restart may be required.")
                
                return True
            else:
                logger.error(f"Translation file not found: {monster_trans_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating description for '{monster_name}': {e}")
            return False
            
    def copy_monster_sprites(self, new_monster_name: str, template_monster: str = "aardorn") -> bool:
        import os
        import shutil
        import glob
        
        try:
            sprites_dir = os.path.join("mods", "tuxemon", "gfx", "sprites")
            battle_dir = os.path.join(sprites_dir, "battle")
            
            if not os.path.exists(battle_dir):
                logger.error(f"Battle sprites directory not found: {battle_dir}")
                return False
                
            template_pattern = os.path.join(battle_dir, f"{template_monster}-*.png")
            template_files = glob.glob(template_pattern)
            
            if not template_files:
                logger.error(f"No sprite files found for template monster '{template_monster}'")
                return False
                
            copied_count = 0
            for template_file in template_files:
                # Skip the front image
                if template_file.endswith(f"{template_monster}-front.png"):
                    logger.error(f"Skipping front image: {template_file}")
                    continue
                    
                filename = os.path.basename(template_file)
                pattern = filename.replace(template_monster, "").lstrip("-")
                
                new_filename = f"{new_monster_name}-{pattern}"
                new_filepath = os.path.join(battle_dir, new_filename)
                
                shutil.copy2(template_file, new_filepath)
                logger.error(f"Copied sprite: {filename} → {new_filename}")
                copied_count += 1
                
            
            logger.error(f"Successfully copied {copied_count} sprite files for '{new_monster_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error copying sprite files for '{new_monster_name}': {e}")
            return False
            
    def clean_encounter_file(self, route_name: str, keep_monster: str) -> bool:
        
        import json
        import os
        
        try:
            encounter_file_path = os.path.join("mods", "tuxemon", "db", "encounter", f"{route_name}.json")
            
            if not os.path.exists(encounter_file_path):
                logger.error(f"Encounter file not found: {encounter_file_path}")
                return False
                
            with open(encounter_file_path, 'r') as f:
                encounter_data = json.load(f)
                
            original_count = len(encounter_data["monsters"])
            
            filtered_monsters = [
                monster for monster in encounter_data["monsters"] 
                if monster["monster"] == keep_monster
            ]
            
            encounter_data["monsters"] = filtered_monsters
            
            with open(encounter_file_path, 'w') as f:
                json.dump(encounter_data, f, indent=2)
                
            kept_count = len(filtered_monsters)
            removed_count = original_count - kept_count
            
            logger.error(f"Cleaned encounter file '{route_name}': kept {kept_count} entries for '{keep_monster}', removed {removed_count} other entries")
            
            from tuxemon.db import db
            db.reload("encounter")
            
            return True
                
        except Exception as e:
            logger.error(f"Error cleaning encounter file: {e}")
            return False
    
    def update(self, world_state: Any) -> None:
        """
        Updates the background thread with the player's current location.
        
        Parameters:
            world_state: The current world state.
        """
        try:
            if hasattr(world_state, 'current_map') and hasattr(world_state.current_map, 'filename'):
                new_map = world_state.current_map.filename
                
                if new_map != self._current_map:
                    self._current_map = new_map
                    logger.error(f"Player moved to: {self._current_map}")
                    
                    
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
            self._thread.daemon = True
            self._thread.start()
            logger.error("Started background monitoring thread")
    
    def stop_thread(self) -> None:
        """
        Stops the background thread if it's running.
        """
        if self._running:
            self._running = False
            logger.error("Stopping background monitoring thread")