# Chapter 14: Integration with Unity

> *"We've built an incredible library of AI-generated sound assets for our game, but now we need to integrate them into Unity properly. How can we create a system that lets us easily import, organize, and manage these assets within our game engine?"*  
> — Technical Sound Designer, Independent Game Studio

## Learning Objectives

In this chapter, you'll learn how to:

- Create efficient workflows for integrating AudioCraft-generated assets into Unity
- Implement sound asset organization systems within Unity projects
- Develop metadata-driven sound management tools
- Build runtime sound selection and playback systems
- Optimize audio performance in Unity with AI-generated content

## Introduction

Creating sound assets is only part of the game development process. Integrating these assets effectively into a game engine like Unity requires careful planning and systematic organization. In this chapter, we'll explore how to build robust systems that bridge the gap between AudioCraft's generation capabilities and Unity's runtime environment.

Unity provides powerful audio systems, but managing large collections of AI-generated assets requires specialized approaches. We'll create custom tools and workflows that handle the complexities of working with varied, procedurally generated audio content.

## Implementation: Unity Asset Integration Pipeline

Let's start by creating a system to prepare and import our AudioCraft-generated assets into Unity. This process involves converting, organizing, and adding metadata to our assets:

```python
import os
import json
import shutil
import subprocess
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field

@dataclass
class UnityAssetMetadata:
    """Metadata for Unity audio assets."""
    
    # Basic information
    asset_id: str
    original_asset_id: str
    display_name: str
    description: str
    
    # Unity-specific properties
    is_loop: bool = False
    volume: float = 1.0
    priority: int = 128
    spatial_blend: float = 0.0  # 0 = 2D, 1 = 3D
    
    # Categorization
    asset_type: str
    category: str
    subcategory: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Organization
    bank: Optional[str] = None
    mixer_group: Optional[str] = None
    
    # Generation info
    prompt: str
    generation_params: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}

class UnityAssetProcessor:
    """
    Process AudioCraft-generated assets for Unity integration.
    
    This system handles the conversion, organization, and metadata
    preparation for importing AudioCraft assets into Unity projects.
    """
    
    def __init__(
        self, 
        source_assets_dir: str,
        unity_project_dir: str,
        unity_assets_path: str = "Assets/Audio",
        ffmpeg_path: str = "ffmpeg"
    ):
        """
        Initialize the Unity asset processor.
        
        Args:
            source_assets_dir: Directory containing AudioCraft assets
            unity_project_dir: Root directory of Unity project
            unity_assets_path: Path within Unity project for assets
            ffmpeg_path: Path to ffmpeg executable
        """
        self.source_dir = source_assets_dir
        self.unity_project_dir = unity_project_dir
        self.unity_assets_path = unity_assets_path
        self.ffmpeg_path = ffmpeg_path
        
        # Full path to Unity audio directory
        self.unity_audio_dir = os.path.join(unity_project_dir, unity_assets_path)
        
        # Create Unity audio directories
        os.makedirs(self.unity_audio_dir, exist_ok=True)
        
        # Load source asset database
        self.source_db_path = os.path.join(source_assets_dir, "asset_database.json")
        if os.path.exists(self.source_db_path):
            with open(self.source_db_path, 'r') as f:
                self.source_database = json.load(f)
        else:
            self.source_database = {}
        
        # Initialize Unity asset database
        self.unity_db_path = os.path.join(self.unity_audio_dir, "audio_assets.json")
        if os.path.exists(self.unity_db_path):
            with open(self.unity_db_path, 'r') as f:
                self.unity_database = json.load(f)
        else:
            self.unity_database = {}
    
    def save_unity_database(self):
        """Save the Unity asset database."""
        with open(self.unity_db_path, 'w') as f:
            json.dump(self.unity_database, f, indent=2)
    
    def convert_audio_for_unity(
        self,
        source_path: str,
        output_path: str,
        normalize: bool = True,
        target_format: str = "wav",
        sample_rate: int = 44100
    ) -> bool:
        """
        Convert audio file to Unity-compatible format.
        
        Args:
            source_path: Path to source audio file
            output_path: Path for converted output
            normalize: Whether to normalize audio levels
            target_format: Output audio format
            sample_rate: Target sample rate
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build ffmpeg command
            cmd = [
                self.ffmpeg_path,
                "-i", source_path,
                "-ar", str(sample_rate)
            ]
            
            # Add normalization if requested
            if normalize:
                cmd.extend(["-af", "loudnorm=I=-16:LRA=11:TP=-1.5"])
            
            # Add output path
            cmd.append(output_path)
            
            # Run conversion
            subprocess.run(cmd, check=True, capture_output=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting {source_path}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def process_asset(
        self,
        asset_id: str,
        display_name: Optional[str] = None,
        is_loop: bool = False,
        spatial_blend: float = 0.0,
        bank: Optional[str] = None,
        mixer_group: Optional[str] = None,
        additional_tags: List[str] = None,
        normalize_audio: bool = True
    ) -> Optional[str]:
        """
        Process a single audio asset for Unity.
        
        Args:
            asset_id: ID of source asset to process
            display_name: Name to display in Unity
            is_loop: Whether the audio should loop
            spatial_blend: 2D to 3D blend (0-1)
            bank: Optional audio bank name
            mixer_group: Optional audio mixer group
            additional_tags: Additional tags to add
            normalize_audio: Whether to normalize audio levels
            
        Returns:
            Unity asset ID if successful, None otherwise
        """
        # Check if asset exists in source database
        if asset_id not in self.source_database:
            print(f"Asset {asset_id} not found in source database")
            return None
        
        # Get source asset info
        source_asset = self.source_database[asset_id]
        source_file = source_asset["file_path"]
        source_metadata = source_asset["metadata"]
        
        # Create Unity asset ID
        unity_asset_id = f"unity_{asset_id}"
        
        # Create display name if not provided
        if not display_name:
            # Use original description, truncated and cleaned up
            display_name = source_metadata["description"]
            if len(display_name) > 50:
                display_name = display_name[:47] + "..."
        
        # Create Unity directory structure
        asset_type = source_metadata["asset_type"]
        category = source_metadata["category"]
        subcategory = source_metadata.get("subcategory")
        
        # Build target path
        target_dir = os.path.join(self.unity_audio_dir, asset_type, category)
        if subcategory:
            target_dir = os.path.join(target_dir, subcategory)
        os.makedirs(target_dir, exist_ok=True)
        
        # Create target file path with sensible naming
        safe_name = display_name.lower().replace(" ", "_")
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        target_file = os.path.join(target_dir, f"{safe_name}_{asset_id[-8:]}.wav")
        
        # Convert audio file
        if not self.convert_audio_for_unity(
            source_path=source_file,
            output_path=target_file,
            normalize=normalize_audio
        ):
            return None
        
        # Create Unity metadata
        tags = list(source_metadata.get("tags", []))
        if additional_tags:
            tags.extend(additional_tags)
        
        unity_metadata = UnityAssetMetadata(
            asset_id=unity_asset_id,
            original_asset_id=asset_id,
            display_name=display_name,
            description=source_metadata.get("description", ""),
            is_loop=is_loop,
            volume=1.0,
            priority=128,
            spatial_blend=spatial_blend,
            asset_type=asset_type,
            category=category,
            subcategory=subcategory,
            tags=tags,
            bank=bank,
            mixer_group=mixer_group,
            prompt=source_metadata.get("prompt", ""),
            generation_params=source_metadata.get("generation_params", {})
        )
        
        # Save metadata file (used for tool integrations)
        meta_file = f"{target_file}.meta.json"
        with open(meta_file, 'w') as f:
            json.dump(unity_metadata.to_dict(), f, indent=2)
        
        # Add to Unity database
        self.unity_database[unity_asset_id] = {
            "metadata": unity_metadata.to_dict(),
            "file_path": os.path.relpath(target_file, self.unity_project_dir),
            "meta_path": os.path.relpath(meta_file, self.unity_project_dir)
        }
        self.save_unity_database()
        
        print(f"Processed asset {unity_asset_id} to {target_file}")
        return unity_asset_id
    
    def batch_process_assets(
        self,
        asset_configs: List[Dict]
    ) -> List[str]:
        """
        Process multiple assets in a batch.
        
        Args:
            asset_configs: List of dictionaries with asset configurations
            
        Returns:
            List of processed Unity asset IDs
        """
        processed_ids = []
        
        for config in asset_configs:
            asset_id = self.process_asset(**config)
            if asset_id:
                processed_ids.append(asset_id)
        
        return processed_ids
    
    def process_scene_assets(
        self,
        scene_id: str,
        scene_dir: str,
        is_loop_ambience: bool = True,
        normalize_ambience: bool = True,
        normalize_music: bool = True,
        normalize_sfx: bool = True
    ) -> Dict[str, List[str]]:
        """
        Process all assets for a complete game scene.
        
        Args:
            scene_id: Scene identifier
            scene_dir: Directory containing scene definition
            is_loop_ambience: Whether ambient sounds should loop
            normalize_ambience: Whether to normalize ambient sounds
            normalize_music: Whether to normalize music
            normalize_sfx: Whether to normalize sound effects
            
        Returns:
            Dictionary mapping asset types to lists of Unity asset IDs
        """
        scene_file = os.path.join(scene_dir, f"{scene_id}.json")
        if not os.path.exists(scene_file):
            print(f"Scene file {scene_file} not found")
            return {}
        
        # Load scene definition
        with open(scene_file, 'r') as f:
            scene = json.load(f)
        
        processed_assets = {
            "ambience": [],
            "music": [],
            "sfx": []
        }
        
        # Process ambience assets
        if "ambience" in scene and scene["ambience"]:
            for ambience_id, ambience in scene["ambience"].items():
                if "asset_id" in ambience and ambience["asset_id"]:
                    unity_id = self.process_asset(
                        asset_id=ambience["asset_id"],
                        display_name=f"{scene['scene_name']} - {ambience_id}",
                        is_loop=is_loop_ambience,
                        spatial_blend=1.0,  # 3D sound
                        bank="Ambience",
                        mixer_group="Ambience",
                        additional_tags=["scene_ambience", scene_id],
                        normalize_audio=normalize_ambience
                    )
                    if unity_id:
                        processed_assets["ambience"].append(unity_id)
        
        # Process music asset
        if "music" in scene and scene["music"] and "asset_id" in scene["music"]:
            unity_id = self.process_asset(
                asset_id=scene["music"]["asset_id"],
                display_name=f"{scene['scene_name']} - Music",
                is_loop=True,
                spatial_blend=0.0,  # 2D sound
                bank="Music",
                mixer_group="Music",
                additional_tags=["scene_music", scene_id],
                normalize_audio=normalize_music
            )
            if unity_id:
                processed_assets["music"].append(unity_id)
        
        # Process sound effects
        if "sound_effects" in scene and scene["sound_effects"]:
            for sfx_id, sfx in scene["sound_effects"].items():
                if "asset_id" in sfx and sfx["asset_id"]:
                    unity_id = self.process_asset(
                        asset_id=sfx["asset_id"],
                        display_name=f"{scene['scene_name']} - {sfx_id}",
                        is_loop=False,
                        spatial_blend=0.8,  # Mostly 3D
                        bank="SFX",
                        mixer_group="SFX",
                        additional_tags=["scene_sfx", scene_id, sfx.get("category", "")],
                        normalize_audio=normalize_sfx
                    )
                    if unity_id:
                        processed_assets["sfx"].append(unity_id)
        
        return processed_assets
    
    def generate_unity_asset_catalog(self, output_file=None):
        """
        Generate a catalog of all Unity audio assets for easy reference.
        
        Args:
            output_file: Path to the output catalog file
            
        Returns:
            Path to the generated catalog
        """
        if output_file is None:
            output_file = os.path.join(self.unity_audio_dir, "AudioAssetCatalog.md")
        
        with open(output_file, 'w') as f:
            f.write("# Unity Audio Asset Catalog\n\n")
            
            # Group assets by type and category
            assets_by_type = {}
            
            for asset_id, asset_info in self.unity_database.items():
                metadata = asset_info["metadata"]
                asset_type = metadata["asset_type"]
                category = metadata["category"]
                subcategory = metadata.get("subcategory", "")
                
                # Initialize dictionaries if needed
                if asset_type not in assets_by_type:
                    assets_by_type[asset_type] = {}
                
                if category not in assets_by_type[asset_type]:
                    assets_by_type[asset_type][category] = {}
                
                if subcategory not in assets_by_type[asset_type][category]:
                    assets_by_type[asset_type][category][subcategory] = []
                
                # Add asset to the appropriate group
                assets_by_type[asset_type][category][subcategory].append({
                    "id": asset_id,
                    "display_name": metadata["display_name"],
                    "file_path": asset_info["file_path"],
                    "description": metadata["description"],
                    "is_loop": metadata["is_loop"],
                    "spatial_blend": metadata["spatial_blend"],
                    "tags": metadata["tags"]
                })
            
            # Write catalog by type and category
            for asset_type, categories in assets_by_type.items():
                f.write(f"## {asset_type.upper()}\n\n")
                
                for category, subcategories in categories.items():
                    f.write(f"### {category}\n\n")
                    
                    for subcategory, assets in subcategories.items():
                        if subcategory:
                            f.write(f"#### {subcategory}\n\n")
                        
                        f.write("| Name | Description | Loop | Spatial | Tags | File |\n")
                        f.write("|---|---|:---:|:---:|---|---|\n")
                        
                        for asset in assets:
                            # Format the tag list
                            tag_str = ", ".join(asset["tags"])
                            
                            # Format true/false as checkboxes
                            loop_str = "✓" if asset["is_loop"] else ""
                            
                            # Format spatial blend as percentage
                            spatial_str = f"{int(asset['spatial_blend'] * 100)}%"
                            
                            f.write(f"| {asset['display_name']} | {asset['description'][:50]}{'...' if len(asset['description']) > 50 else ''} | {loop_str} | {spatial_str} | {tag_str} | {asset['file_path']} |\n")
                        
                        f.write("\n")
            
            f.write("\n## Usage in Unity\n\n")
            f.write("This catalog lists all audio assets processed for Unity integration. To use these assets in your project:\n\n")
            f.write("1. Import the assets through Unity's Asset menu or by placing them in your project's Assets folder\n")
            f.write("2. Use the Asset Database to look up assets by type, category, or tags\n")
            f.write("3. Attach audio files to AudioSource components on game objects\n")
            f.write("4. For procedural audio selection, use the AudioManager script with the asset tags\n\n")
        
        print(f"Generated Unity asset catalog at {output_file}")
        return output_file

    def generate_c_sharp_constants(self, output_file=None):
        """
        Generate C# constants file for Unity integration.
        
        Args:
            output_file: Path to the output C# file
            
        Returns:
            Path to the generated file
        """
        if output_file is None:
            output_file = os.path.join(
                self.unity_project_dir, 
                "Assets/Scripts/Audio", 
                "AudioAssetConstants.cs"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("// Auto-generated audio asset constants\n")
            f.write("// Do not modify this file manually\n\n")
            f.write("namespace GameAudio {\n\n")
            
            # Audio asset IDs
            f.write("    public static class AudioAssets {\n")
            
            # Group by type and category
            assets_by_type = {}
            
            for asset_id, asset_info in self.unity_database.items():
                metadata = asset_info["metadata"]
                asset_type = metadata["asset_type"]
                
                if asset_type not in assets_by_type:
                    assets_by_type[asset_type] = []
                
                # Clean up the display name for constant creation
                display_name = metadata["display_name"]
                constant_name = ''.join(c for c in display_name if c.isalnum() or c == ' ')
                constant_name = ''.join(word.capitalize() for word in constant_name.split())
                
                assets_by_type[asset_type].append({
                    "id": asset_id,
                    "constant_name": constant_name,
                    "display_name": display_name
                })
            
            # Write constants by type
            for asset_type, assets in assets_by_type.items():
                f.write(f"\n        // {asset_type.capitalize()} assets\n")
                
                for asset in assets:
                    f.write(f"        public const string {asset['constant_name']} = \"{asset['id']}\";\n")
            
            f.write("    }\n\n")
            
            # Audio tags
            f.write("    public static class AudioTags {\n")
            
            # Collect all unique tags
            all_tags = set()
            for asset_info in self.unity_database.values():
                metadata = asset_info["metadata"]
                if "tags" in metadata:
                    all_tags.update(metadata["tags"])
            
            # Write tag constants
            for tag in sorted(all_tags):
                tag_constant = ''.join(word.capitalize() for word in tag.split('_'))
                tag_constant = ''.join(c for c in tag_constant if c.isalnum())
                f.write(f"        public const string {tag_constant} = \"{tag}\";\n")
            
            f.write("    }\n\n")
            
            # Audio categories
            f.write("    public static class AudioCategories {\n")
            
            # Collect all unique categories
            categories = set()
            for asset_info in self.unity_database.values():
                metadata = asset_info["metadata"]
                categories.add(metadata["category"])
                if metadata.get("subcategory"):
                    categories.add(metadata["subcategory"])
            
            # Write category constants
            for category in sorted(categories):
                category_constant = ''.join(word.capitalize() for word in category.split('_'))
                category_constant = ''.join(c for c in category_constant if c.isalnum())
                f.write(f"        public const string {category_constant} = \"{category}\";\n")
            
            f.write("    }\n")
            f.write("}\n")
        
        print(f"Generated C# constants file at {output_file}")
        return output_file
```

## Unity Audio Manager Implementation

Let's create a complementary Unity C# script that will manage our AudioCraft-generated audio assets at runtime:

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Audio;

namespace GameAudio
{
    /// <summary>
    /// Manages audio assets and provides tools for runtime audio selection and playback.
    /// This system integrates with AudioCraft-generated content to provide rich,
    /// metadata-driven audio playback for game projects.
    /// </summary>
    public class AudioManager : MonoBehaviour
    {
        // Singleton instance
        public static AudioManager Instance { get; private set; }

        [Serializable]
        public class AudioAssetData
        {
            public string assetId;
            public AudioClip clip;
            public bool isLoop;
            [Range(0f, 1f)] public float spatialBlend;
            [Range(0f, 1f)] public float volume = 1f;
            public int priority = 128;
            public string[] tags;
            public string category;
            public string subcategory;
            public string bank;
            public string mixerGroup;
        }

        [Header("Audio Assets")]
        [SerializeField] private List<AudioAssetData> audioAssets = new List<AudioAssetData>();

        [Header("Audio Settings")]
        [SerializeField] private AudioMixerGroup masterMixerGroup;
        [SerializeField] private AudioMixerGroup musicMixerGroup;
        [SerializeField] private AudioMixerGroup sfxMixerGroup;
        [SerializeField] private AudioMixerGroup ambienceMixerGroup;
        [SerializeField] private AudioMixerGroup voiceMixerGroup;
        [SerializeField] private AudioMixerGroup uiMixerGroup;
        
        [Header("Pool Settings")]
        [SerializeField] private int initialPoolSize = 10;
        [SerializeField] private int maxPoolSize = 20;
        
        // Pooled audio sources
        private List<AudioSource> pooledSources = new List<AudioSource>();
        
        // Active sounds
        private Dictionary<string, AudioSource> activeSounds = new Dictionary<string, AudioSource>();
        
        // Asset lookup dictionaries
        private Dictionary<string, AudioAssetData> assetsById = new Dictionary<string, AudioAssetData>();
        private Dictionary<string, List<AudioAssetData>> assetsByTag = new Dictionary<string, List<AudioAssetData>>();
        private Dictionary<string, List<AudioAssetData>> assetsByCategory = new Dictionary<string, List<AudioAssetData>>();
        
        private void Awake()
        {
            // Singleton pattern
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            
            Instance = this;
            DontDestroyOnLoad(gameObject);
            
            // Initialize lookup dictionaries
            InitializeAssetLookups();
            
            // Initialize audio source pool
            InitializeSourcePool();
        }
        
        private void InitializeAssetLookups()
        {
            // Create lookup dictionaries for efficient asset retrieval
            foreach (var asset in audioAssets)
            {
                // Add to ID lookup
                assetsById[asset.assetId] = asset;
                
                // Add to tag lookup
                if (asset.tags != null)
                {
                    foreach (var tag in asset.tags)
                    {
                        if (!assetsByTag.ContainsKey(tag))
                        {
                            assetsByTag[tag] = new List<AudioAssetData>();
                        }
                        assetsByTag[tag].Add(asset);
                    }
                }
                
                // Add to category lookup
                if (!string.IsNullOrEmpty(asset.category))
                {
                    if (!assetsByCategory.ContainsKey(asset.category))
                    {
                        assetsByCategory[asset.category] = new List<AudioAssetData>();
                    }
                    assetsByCategory[asset.category].Add(asset);
                }
                
                // Add to subcategory lookup
                if (!string.IsNullOrEmpty(asset.subcategory))
                {
                    if (!assetsByCategory.ContainsKey(asset.subcategory))
                    {
                        assetsByCategory[asset.subcategory] = new List<AudioAssetData>();
                    }
                    assetsByCategory[asset.subcategory].Add(asset);
                }
            }
            
            Debug.Log($"Audio Manager initialized with {audioAssets.Count} assets");
        }
        
        private void InitializeSourcePool()
        {
            // Create initial pool of audio sources
            for (int i = 0; i < initialPoolSize; i++)
            {
                CreatePooledAudioSource();
            }
        }
        
        private AudioSource CreatePooledAudioSource()
        {
            // Create a new game object for the audio source
            GameObject sourceObj = new GameObject("Pooled Audio Source");
            sourceObj.transform.SetParent(transform);
            
            // Add audio source component
            AudioSource source = sourceObj.AddComponent<AudioSource>();
            source.playOnAwake = false;
            
            // Add to pool
            pooledSources.Add(source);
            
            return source;
        }
        
        private AudioSource GetAudioSourceFromPool()
        {
            // Look for an inactive audio source in the pool
            foreach (var source in pooledSources)
            {
                if (!source.isPlaying)
                {
                    return source;
                }
            }
            
            // If all sources are in use and we haven't reached the max pool size,
            // create a new one
            if (pooledSources.Count < maxPoolSize)
            {
                return CreatePooledAudioSource();
            }
            
            // If we've reached the max pool size, find the lowest priority sound
            AudioSource lowestPrioritySource = pooledSources[0];
            int lowestPriority = lowestPrioritySource.priority;
            
            foreach (var source in pooledSources)
            {
                if (source.priority < lowestPriority)
                {
                    lowestPrioritySource = source;
                    lowestPriority = source.priority;
                }
            }
            
            // Stop the lowest priority sound and return its source
            lowestPrioritySource.Stop();
            return lowestPrioritySource;
        }
        
        /// <summary>
        /// Get the appropriate mixer group for an asset type
        /// </summary>
        private AudioMixerGroup GetMixerGroupForAsset(AudioAssetData asset)
        {
            if (!string.IsNullOrEmpty(asset.mixerGroup))
            {
                switch (asset.mixerGroup.ToLower())
                {
                    case "music":
                        return musicMixerGroup;
                    case "sfx":
                        return sfxMixerGroup;
                    case "ambience":
                        return ambienceMixerGroup;
                    case "voice":
                        return voiceMixerGroup;
                    case "ui":
                        return uiMixerGroup;
                }
            }
            
            // Default to master group
            return masterMixerGroup;
        }
        
        /// <summary>
        /// Play an audio asset by its ID
        /// </summary>
        public AudioSource PlaySound(string assetId, Vector3? position = null, float volumeMultiplier = 1.0f)
        {
            if (!assetsById.TryGetValue(assetId, out AudioAssetData asset))
            {
                Debug.LogWarning($"Audio asset not found: {assetId}");
                return null;
            }
            
            // Get an audio source from the pool
            AudioSource source = GetAudioSourceFromPool();
            
            // Configure the audio source
            source.clip = asset.clip;
            source.loop = asset.isLoop;
            source.spatialBlend = asset.spatialBlend;
            source.volume = asset.volume * volumeMultiplier;
            source.priority = asset.priority;
            source.outputAudioMixerGroup = GetMixerGroupForAsset(asset);
            
            // Set position if provided
            if (position.HasValue)
            {
                source.transform.position = position.Value;
            }
            
            // Play the sound
            source.Play();
            
            // If looping, add to active sounds dictionary
            if (asset.isLoop)
            {
                string playbackId = $"{assetId}_{DateTime.Now.Ticks}";
                activeSounds[playbackId] = source;
                return source;
            }
            
            return source;
        }
        
        /// <summary>
        /// Play a random audio asset with the specified tag
        /// </summary>
        public AudioSource PlayRandomSound(string tag, Vector3? position = null, float volumeMultiplier = 1.0f)
        {
            if (!assetsByTag.TryGetValue(tag, out List<AudioAssetData> matchingAssets) || matchingAssets.Count == 0)
            {
                Debug.LogWarning($"No audio assets found with tag: {tag}");
                return null;
            }
            
            // Select a random asset
            AudioAssetData asset = matchingAssets[UnityEngine.Random.Range(0, matchingAssets.Count)];
            
            // Play the selected asset
            return PlaySound(asset.assetId, position, volumeMultiplier);
        }
        
        /// <summary>
        /// Play a random audio asset with all the specified tags (AND logic)
        /// </summary>
        public AudioSource PlayRandomSoundWithAllTags(string[] tags, Vector3? position = null, float volumeMultiplier = 1.0f)
        {
            if (tags == null || tags.Length == 0)
            {
                Debug.LogWarning("No tags specified for PlayRandomSoundWithAllTags");
                return null;
            }
            
            // Find assets that have all the specified tags
            List<AudioAssetData> candidates = new List<AudioAssetData>();
            
            foreach (var asset in audioAssets)
            {
                bool hasAllTags = true;
                
                foreach (var tag in tags)
                {
                    if (asset.tags == null || !asset.tags.Contains(tag))
                    {
                        hasAllTags = false;
                        break;
                    }
                }
                
                if (hasAllTags)
                {
                    candidates.Add(asset);
                }
            }
            
            if (candidates.Count == 0)
            {
                Debug.LogWarning($"No audio assets found with all tags: {string.Join(", ", tags)}");
                return null;
            }
            
            // Select a random asset from candidates
            AudioAssetData selectedAsset = candidates[UnityEngine.Random.Range(0, candidates.Count)];
            
            // Play the selected asset
            return PlaySound(selectedAsset.assetId, position, volumeMultiplier);
        }
        
        /// <summary>
        /// Play a random audio asset with any of the specified tags (OR logic)
        /// </summary>
        public AudioSource PlayRandomSoundWithAnyTag(string[] tags, Vector3? position = null, float volumeMultiplier = 1.0f)
        {
            if (tags == null || tags.Length == 0)
            {
                Debug.LogWarning("No tags specified for PlayRandomSoundWithAnyTag");
                return null;
            }
            
            // Find assets that have any of the specified tags
            HashSet<AudioAssetData> candidates = new HashSet<AudioAssetData>();
            
            foreach (var tag in tags)
            {
                if (assetsByTag.TryGetValue(tag, out List<AudioAssetData> matchingAssets))
                {
                    foreach (var asset in matchingAssets)
                    {
                        candidates.Add(asset);
                    }
                }
            }
            
            if (candidates.Count == 0)
            {
                Debug.LogWarning($"No audio assets found with any tag: {string.Join(", ", tags)}");
                return null;
            }
            
            // Select a random asset from candidates
            AudioAssetData selectedAsset = candidates.ElementAt(UnityEngine.Random.Range(0, candidates.Count));
            
            // Play the selected asset
            return PlaySound(selectedAsset.assetId, position, volumeMultiplier);
        }
        
        /// <summary>
        /// Play a random audio asset from the specified category
        /// </summary>
        public AudioSource PlayRandomSoundFromCategory(string category, Vector3? position = null, float volumeMultiplier = 1.0f)
        {
            if (!assetsByCategory.TryGetValue(category, out List<AudioAssetData> matchingAssets) || matchingAssets.Count == 0)
            {
                Debug.LogWarning($"No audio assets found in category: {category}");
                return null;
            }
            
            // Select a random asset
            AudioAssetData asset = matchingAssets[UnityEngine.Random.Range(0, matchingAssets.Count)];
            
            // Play the selected asset
            return PlaySound(asset.assetId, position, volumeMultiplier);
        }
        
        /// <summary>
        /// Stop all sounds with the specified tag
        /// </summary>
        public void StopSoundsWithTag(string tag)
        {
            if (!assetsByTag.TryGetValue(tag, out List<AudioAssetData> matchingAssets))
            {
                return;
            }
            
            // Get IDs of matching assets
            HashSet<string> matchingIds = new HashSet<string>(matchingAssets.Select(a => a.assetId));
            
            // Find active sounds with matching asset IDs
            List<string> keysToRemove = new List<string>();
            
            foreach (var kvp in activeSounds)
            {
                string playbackId = kvp.Key;
                AudioSource source = kvp.Value;
                
                // Extract the asset ID from the playback ID
                string assetId = playbackId.Substring(0, playbackId.LastIndexOf('_'));
                
                if (matchingIds.Contains(assetId))
                {
                    source.Stop();
                    keysToRemove.Add(playbackId);
                }
            }
            
            // Remove stopped sounds from active sounds dictionary
            foreach (var key in keysToRemove)
            {
                activeSounds.Remove(key);
            }
        }
        
        /// <summary>
        /// Stop all sounds in the specified category
        /// </summary>
        public void StopSoundsInCategory(string category)
        {
            if (!assetsByCategory.TryGetValue(category, out List<AudioAssetData> matchingAssets))
            {
                return;
            }
            
            // Get IDs of matching assets
            HashSet<string> matchingIds = new HashSet<string>(matchingAssets.Select(a => a.assetId));
            
            // Find active sounds with matching asset IDs
            List<string> keysToRemove = new List<string>();
            
            foreach (var kvp in activeSounds)
            {
                string playbackId = kvp.Key;
                AudioSource source = kvp.Value;
                
                // Extract the asset ID from the playback ID
                string assetId = playbackId.Substring(0, playbackId.LastIndexOf('_'));
                
                if (matchingIds.Contains(assetId))
                {
                    source.Stop();
                    keysToRemove.Add(playbackId);
                }
            }
            
            // Remove stopped sounds from active sounds dictionary
            foreach (var key in keysToRemove)
            {
                activeSounds.Remove(key);
            }
        }
        
        /// <summary>
        /// Stop all sounds
        /// </summary>
        public void StopAllSounds()
        {
            foreach (var source in pooledSources)
            {
                source.Stop();
            }
            
            activeSounds.Clear();
        }
        
        /// <summary>
        /// Fade in a sound over time
        /// </summary>
        public AudioSource FadeInSound(string assetId, float fadeDuration, Vector3? position = null)
        {
            // Play the sound at zero volume
            AudioSource source = PlaySound(assetId, position, 0);
            
            if (source != null)
            {
                // Start coroutine to fade in
                StartCoroutine(FadeAudioSource(source, 0, source.volume, fadeDuration));
            }
            
            return source;
        }
        
        /// <summary>
        /// Fade out and stop a sound over time
        /// </summary>
        public void FadeOutSound(AudioSource source, float fadeDuration)
        {
            if (source != null && source.isPlaying)
            {
                // Start coroutine to fade out
                StartCoroutine(FadeAudioSource(source, source.volume, 0, fadeDuration, true));
            }
        }
        
        /// <summary>
        /// Coroutine to fade audio source volume
        /// </summary>
        private System.Collections.IEnumerator FadeAudioSource(
            AudioSource source, float startVolume, float targetVolume, 
            float duration, bool stopAfterFade = false)
        {
            float currentTime = 0;
            source.volume = startVolume;
            
            while (currentTime < duration)
            {
                currentTime += Time.deltaTime;
                source.volume = Mathf.Lerp(startVolume, targetVolume, currentTime / duration);
                yield return null;
            }
            
            // Ensure we end at the exact target volume
            source.volume = targetVolume;
            
            // Stop the source if requested
            if (stopAfterFade && source.isPlaying)
            {
                source.Stop();
                
                // Remove from active sounds if present
                foreach (var kvp in activeSounds.ToList())
                {
                    if (kvp.Value == source)
                    {
                        activeSounds.Remove(kvp.Key);
                        break;
                    }
                }
            }
        }
        
        /// <summary>
        /// Get all audio assets with the specified tag
        /// </summary>
        public List<AudioAssetData> GetAssetsWithTag(string tag)
        {
            if (assetsByTag.TryGetValue(tag, out List<AudioAssetData> assets))
            {
                return new List<AudioAssetData>(assets);
            }
            
            return new List<AudioAssetData>();
        }
        
        /// <summary>
        /// Get all audio assets in the specified category
        /// </summary>
        public List<AudioAssetData> GetAssetsInCategory(string category)
        {
            if (assetsByCategory.TryGetValue(category, out List<AudioAssetData> assets))
            {
                return new List<AudioAssetData>(assets);
            }
            
            return new List<AudioAssetData>();
        }
    }
    
    /// <summary>
    /// Custom editor for AudioManager to simplify asset assignment and management
    /// </summary>
    #if UNITY_EDITOR
    [UnityEditor.CustomEditor(typeof(AudioManager))]
    public class AudioManagerEditor : UnityEditor.Editor
    {
        private bool showAssetList = true;
        private string searchFilter = "";
        private string categoryFilter = "";
        private string tagFilter = "";
        
        public override void OnInspectorGUI()
        {
            UnityEditor.EditorGUILayout.LabelField("Audio Manager", UnityEditor.EditorStyles.boldLabel);
            UnityEditor.EditorGUILayout.Space();
            
            // Draw default inspector properties
            DrawDefaultInspector();
            
            AudioManager manager = (AudioManager)target;
            
            UnityEditor.EditorGUILayout.Space();
            UnityEditor.EditorGUILayout.LabelField("Asset List", UnityEditor.EditorStyles.boldLabel);
            
            // Search filters
            UnityEditor.EditorGUILayout.BeginHorizontal();
            UnityEditor.EditorGUILayout.LabelField("Search:", GUILayout.Width(60));
            searchFilter = UnityEditor.EditorGUILayout.TextField(searchFilter);
            UnityEditor.EditorGUILayout.EndHorizontal();
            
            UnityEditor.EditorGUILayout.BeginHorizontal();
            UnityEditor.EditorGUILayout.LabelField("Category:", GUILayout.Width(60));
            categoryFilter = UnityEditor.EditorGUILayout.TextField(categoryFilter);
            UnityEditor.EditorGUILayout.EndHorizontal();
            
            UnityEditor.EditorGUILayout.BeginHorizontal();
            UnityEditor.EditorGUILayout.LabelField("Tag:", GUILayout.Width(60));
            tagFilter = UnityEditor.EditorGUILayout.TextField(tagFilter);
            UnityEditor.EditorGUILayout.EndHorizontal();
            
            UnityEditor.EditorGUILayout.Space();
            
            // Import assets button
            if (GUILayout.Button("Import Audio Assets from JSON"))
            {
                string path = UnityEditor.EditorUtility.OpenFilePanel("Select audio assets JSON", "Assets", "json");
                if (!string.IsNullOrEmpty(path))
                {
                    ImportAssetsFromJson(path);
                }
            }
            
            // Asset list foldout
            showAssetList = UnityEditor.EditorGUILayout.Foldout(showAssetList, "Audio Assets");
            
            if (showAssetList)
            {
                // Get serialized property for audioAssets list
                UnityEditor.SerializedProperty assetListProp = serializedObject.FindProperty("audioAssets");
                
                for (int i = 0; i < assetListProp.arraySize; i++)
                {
                    UnityEditor.SerializedProperty assetProp = assetListProp.GetArrayElementAtIndex(i);
                    UnityEditor.SerializedProperty assetIdProp = assetProp.FindPropertyRelative("assetId");
                    UnityEditor.SerializedProperty clipProp = assetProp.FindPropertyRelative("clip");
                    UnityEditor.SerializedProperty categoryProp = assetProp.FindPropertyRelative("category");
                    UnityEditor.SerializedProperty tagsProp = assetProp.FindPropertyRelative("tags");
                    
                    // Apply filters
                    bool matchesSearch = string.IsNullOrEmpty(searchFilter) || 
                        assetIdProp.stringValue.ToLower().Contains(searchFilter.ToLower());
                    
                    bool matchesCategory = string.IsNullOrEmpty(categoryFilter) ||
                        (categoryProp != null && categoryProp.stringValue.ToLower().Contains(categoryFilter.ToLower()));
                    
                    bool matchesTag = string.IsNullOrEmpty(tagFilter);
                    
                    if (!matchesTag && tagsProp != null)
                    {
                        for (int j = 0; j < tagsProp.arraySize; j++)
                        {
                            string tag = tagsProp.GetArrayElementAtIndex(j).stringValue;
                            if (tag.ToLower().Contains(tagFilter.ToLower()))
                            {
                                matchesTag = true;
                                break;
                            }
                        }
                    }
                    
                    if (matchesSearch && matchesCategory && matchesTag)
                    {
                        UnityEditor.EditorGUILayout.PropertyField(assetProp, new GUIContent(assetIdProp.stringValue), true);
                    }
                }
            }
            
            serializedObject.ApplyModifiedProperties();
        }
        
        private void ImportAssetsFromJson(string jsonPath)
        {
            try
            {
                // Read JSON file
                string jsonText = System.IO.File.ReadAllText(jsonPath);
                Dictionary<string, object> assetDatabase = UnityEngine.JsonUtility.FromJson<Dictionary<string, object>>(jsonText);
                
                // TODO: Parse JSON and populate audio assets
                UnityEditor.EditorUtility.DisplayDialog("Import Assets", 
                    "Asset import from JSON is not fully implemented yet. This would populate the AudioManager with asset data from the JSON file.", "OK");
            }
            catch (System.Exception e)
            {
                UnityEditor.EditorUtility.DisplayDialog("Import Error", $"Error importing assets: {e.Message}", "OK");
            }
        }
    }
    #endif
}
```

## Scene-Based Sound Manager

For more complex game scenarios, let's create a scene-based sound manager that handles environment transitions and layered audio:

```csharp
using System;
using System.Collections.Generic;
using UnityEngine;

namespace GameAudio
{
    /// <summary>
    /// Manages scene-based sound environments, including ambient layers,
    /// music, and scene-specific sound effects.
    /// </summary>
    public class SceneSoundManager : MonoBehaviour
    {
        [Serializable]
        public class SceneAudioData
        {
            public string sceneName;
            
            [Header("Ambient Sounds")]
            public string[] ambienceAssetIds;
            [Range(0f, 1f)] public float[] ambienceVolumes = new float[0];
            
            [Header("Music")]
            public string musicAssetId;
            [Range(0f, 1f)] public float musicVolume = 1.0f;
            
            [Header("Scene Sound Trigger Tags")]
            public string[] soundTriggerTags;
        }
        
        [Header("Scene Sound Definitions")]
        [SerializeField] private List<SceneAudioData> sceneAudioData = new List<SceneAudioData>();
        
        [Header("Transition Settings")]
        [SerializeField] private float crossfadeDuration = 3.0f;
        [SerializeField] private float ambienceFadeInDelay = 0.5f;
        [SerializeField] private float musicFadeInDelay = 2.0f;
        
        // Current scene data
        private SceneAudioData currentSceneData;
        
        // Active audio sources
        private List<AudioSource> activeAmbienceSources = new List<AudioSource>();
        private AudioSource activeMusicSource;
        
        // Cache
        private Dictionary<string, SceneAudioData> sceneDataLookup = new Dictionary<string, SceneAudioData>();
        
        private void Awake()
        {
            // Build scene data lookup
            foreach (var data in sceneAudioData)
            {
                sceneDataLookup[data.sceneName] = data;
            }
        }
        
        private void Start()
        {
            // Get the initial scene
            string initialSceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            
            // Transition to initial scene audio
            TransitionToScene(initialSceneName, 0f);
            
            // Listen for scene changes
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged += OnActiveSceneChanged;
        }
        
        private void OnDestroy()
        {
            // Remove scene change listener
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged -= OnActiveSceneChanged;
        }
        
        private void OnActiveSceneChanged(UnityEngine.SceneManagement.Scene oldScene, UnityEngine.SceneManagement.Scene newScene)
        {
            // Transition to new scene audio
            TransitionToScene(newScene.name);
        }
        
        /// <summary>
        /// Transition to audio for the specified scene
        /// </summary>
        public void TransitionToScene(string sceneName, float? customCrossfadeDuration = null)
        {
            if (!sceneDataLookup.TryGetValue(sceneName, out SceneAudioData newSceneData))
            {
                Debug.LogWarning($"No audio data defined for scene: {sceneName}");
                return;
            }
            
            float fadeDuration = customCrossfadeDuration ?? crossfadeDuration;
            
            // Stop current ambience with fade
            foreach (var source in activeAmbienceSources)
            {
                if (source != null && source.isPlaying)
                {
                    AudioManager.Instance.FadeOutSound(source, fadeDuration);
                }
            }
            
            // Clear ambience list
            activeAmbienceSources.Clear();
            
            // Start new ambience with delay
            StartCoroutine(StartAmbienceWithDelay(newSceneData, fadeDuration, ambienceFadeInDelay));
            
            // Handle music transition
            if (activeMusicSource != null && activeMusicSource.isPlaying)
            {
                // Fade out current music
                AudioManager.Instance.FadeOutSound(activeMusicSource, fadeDuration);
                activeMusicSource = null;
            }
            
            // Start new music with delay
            if (!string.IsNullOrEmpty(newSceneData.musicAssetId))
            {
                StartCoroutine(StartMusicWithDelay(newSceneData, fadeDuration, musicFadeInDelay));
            }
            
            // Update current scene data
            currentSceneData = newSceneData;
        }
        
        private System.Collections.IEnumerator StartAmbienceWithDelay(
            SceneAudioData sceneData, float fadeDuration, float delay)
        {
            // Wait for delay
            yield return new WaitForSeconds(delay);
            
            // Start all ambience layers
            if (sceneData.ambienceAssetIds != null)
            {
                for (int i = 0; i < sceneData.ambienceAssetIds.Length; i++)
                {
                    string assetId = sceneData.ambienceAssetIds[i];
                    
                    if (string.IsNullOrEmpty(assetId))
                        continue;
                    
                    // Get volume for this layer
                    float volume = 1.0f;
                    if (i < sceneData.ambienceVolumes.Length)
                    {
                        volume = sceneData.ambienceVolumes[i];
                    }
                    
                    // Fade in the ambience
                    AudioSource source = AudioManager.Instance.FadeInSound(assetId, fadeDuration);
                    
                    if (source != null)
                    {
                        // Store for later reference
                        activeAmbienceSources.Add(source);
                    }
                }
            }
        }
        
        private System.Collections.IEnumerator StartMusicWithDelay(
            SceneAudioData sceneData, float fadeDuration, float delay)
        {
            // Wait for delay
            yield return new WaitForSeconds(delay);
            
            // Start music
            if (!string.IsNullOrEmpty(sceneData.musicAssetId))
            {
                AudioSource source = AudioManager.Instance.FadeInSound(
                    sceneData.musicAssetId, 
                    fadeDuration, 
                    null  // No position for music (2D)
                );
                
                if (source != null)
                {
                    // Apply music volume
                    source.volume = sceneData.musicVolume;
                    
                    // Store for later reference
                    activeMusicSource = source;
                }
            }
        }
        
        /// <summary>
        /// Play a random sound effect for the current scene with the specified tag
        /// </summary>
        public AudioSource PlayRandomSceneSound(string tag, Vector3? position = null)
        {
            if (currentSceneData == null || currentSceneData.soundTriggerTags == null)
            {
                return null;
            }
            
            // Check if this tag is in the scene's sound trigger tags
            bool isValidTag = false;
            foreach (var sceneTag in currentSceneData.soundTriggerTags)
            {
                if (sceneTag == tag)
                {
                    isValidTag = true;
                    break;
                }
            }
            
            if (!isValidTag)
            {
                Debug.LogWarning($"Tag '{tag}' is not defined for the current scene");
                return null;
            }
            
            // Play a random sound with this tag
            return AudioManager.Instance.PlayRandomSound(tag, position);
        }
        
        /// <summary>
        /// Play a random sound effect with the specified tag, filtered by scene name
        /// </summary>
        public AudioSource PlayRandomSoundForScene(string tag, string sceneName, Vector3? position = null)
        {
            // Create combined tag for scene-specific sounds
            string sceneTag = $"scene_{sceneName}";
            
            // Play a sound with both tags
            return AudioManager.Instance.PlayRandomSoundWithAllTags(
                new string[] { tag, sceneTag },
                position
            );
        }
    }
}
```

## Editor Tool for AudioCraft-Unity Integration

To complete our workflow, let's create an editor tool that streamlines the process of importing AudioCraft assets into Unity:

```csharp
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace GameAudio.Editor
{
    /// <summary>
    /// Editor window for importing and managing AudioCraft-generated assets.
    /// </summary>
    public class AudioCraftImportTool : EditorWindow
    {
        // Settings
        private string sourceDatabasePath = "";
        private string outputDirectory = "Assets/Audio";
        private bool normalizeAudio = true;
        private bool createAudioMixerGroups = true;
        private bool generateConstants = true;
        
        // Asset filters
        private bool importMusic = true;
        private bool importAmbience = true;
        private bool importSFX = true;
        private bool importUI = true;
        private string filterByProject = "";
        private string filterByCategory = "";
        
        // Utility paths
        private string ffmpegPath = "/usr/local/bin/ffmpeg";
        private bool ffmpegFound = false;
        
        // Progress tracking
        private bool isImporting = false;
        private float importProgress = 0f;
        private string statusMessage = "";
        private List<string> logMessages = new List<string>();
        private Vector2 logScrollPosition;
        
        [MenuItem("Tools/AudioCraft Import Tool")]
        public static void ShowWindow()
        {
            AudioCraftImportTool window = GetWindow<AudioCraftImportTool>();
            window.titleContent = new GUIContent("AudioCraft Import");
            window.minSize = new Vector2(500, 600);
            window.Show();
        }
        
        private void OnEnable()
        {
            // Check for ffmpeg
            CheckForFFmpeg();
        }
        
        private void OnGUI()
        {
            GUILayout.Label("AudioCraft Asset Importer", EditorStyles.boldLabel);
            EditorGUILayout.Space();
            
            DrawSourceSettings();
            EditorGUILayout.Space();
            
            DrawImportFilters();
            EditorGUILayout.Space();
            
            DrawUtilitySettings();
            EditorGUILayout.Space();
            
            DrawImportButton();
            EditorGUILayout.Space();
            
            DrawProgressBar();
            EditorGUILayout.Space();
            
            DrawLog();
        }
        
        private void DrawSourceSettings()
        {
            GUILayout.Label("Source Settings", EditorStyles.boldLabel);
            
            EditorGUILayout.BeginHorizontal();
            sourceDatabasePath = EditorGUILayout.TextField("Asset Database Path:", sourceDatabasePath);
            if (GUILayout.Button("Browse", GUILayout.Width(70)))
            {
                string path = EditorUtility.OpenFilePanel("Select AudioCraft Asset Database", "", "json");
                if (!string.IsNullOrEmpty(path))
                {
                    sourceDatabasePath = path;
                }
            }
            EditorGUILayout.EndHorizontal();
            
            EditorGUILayout.BeginHorizontal();
            outputDirectory = EditorGUILayout.TextField("Unity Output Directory:", outputDirectory);
            if (GUILayout.Button("Browse", GUILayout.Width(70)))
            {
                string path = EditorUtility.SaveFolderPanel("Select Unity Output Directory", "Assets", "");
                if (!string.IsNullOrEmpty(path))
                {
                    string relativePath = path;
                    if (path.StartsWith(Application.dataPath))
                    {
                        relativePath = "Assets" + path.Substring(Application.dataPath.Length);
                    }
                    outputDirectory = relativePath;
                }
            }
            EditorGUILayout.EndHorizontal();
            
            normalizeAudio = EditorGUILayout.Toggle("Normalize Audio", normalizeAudio);
            createAudioMixerGroups = EditorGUILayout.Toggle("Create Audio Mixer Groups", createAudioMixerGroups);
            generateConstants = EditorGUILayout.Toggle("Generate C# Constants", generateConstants);
        }
        
        private void DrawImportFilters()
        {
            GUILayout.Label("Import Filters", EditorStyles.boldLabel);
            
            EditorGUILayout.BeginHorizontal();
                importMusic = EditorGUILayout.ToggleLeft("Music", importMusic, GUILayout.Width(80));
                importAmbience = EditorGUILayout.ToggleLeft("Ambience", importAmbience, GUILayout.Width(100));
                importSFX = EditorGUILayout.ToggleLeft("SFX", importSFX, GUILayout.Width(80));
                importUI = EditorGUILayout.ToggleLeft("UI", importUI, GUILayout.Width(80));
            EditorGUILayout.EndHorizontal();
            
            filterByProject = EditorGUILayout.TextField("Filter by Project:", filterByProject);
            filterByCategory = EditorGUILayout.TextField("Filter by Category:", filterByCategory);
        }
        
        private void DrawUtilitySettings()
        {
            GUILayout.Label("Utility Settings", EditorStyles.boldLabel);
            
            EditorGUILayout.BeginHorizontal();
            ffmpegPath = EditorGUILayout.TextField("FFmpeg Path:", ffmpegPath);
            if (GUILayout.Button("Check", GUILayout.Width(70)))
            {
                CheckForFFmpeg();
            }
            EditorGUILayout.EndHorizontal();
            
            if (ffmpegFound)
            {
                EditorGUILayout.HelpBox("FFmpeg found and working properly.", MessageType.Info);
            }
            else
            {
                EditorGUILayout.HelpBox(
                    "FFmpeg not found or not working. Please install FFmpeg and set the correct path.", 
                    MessageType.Warning
                );
            }
        }
        
        private void DrawImportButton()
        {
            EditorGUI.BeginDisabledGroup(isImporting || string.IsNullOrEmpty(sourceDatabasePath) || !ffmpegFound);
            
            if (GUILayout.Button("Import AudioCraft Assets", GUILayout.Height(40)))
            {
                StartImport();
            }
            
            EditorGUI.EndDisabledGroup();
            
            if (!ffmpegFound)
            {
                EditorGUILayout.HelpBox("FFmpeg is required for audio conversion. Please install it before importing.", MessageType.Error);
            }
        }
        
        private void DrawProgressBar()
        {
            if (isImporting)
            {
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(false, 20f), importProgress, statusMessage);
            }
        }
        
        private void DrawLog()
        {
            GUILayout.Label("Import Log", EditorStyles.boldLabel);
            
            logScrollPosition = EditorGUILayout.BeginScrollView(logScrollPosition, GUILayout.Height(200));
            
            foreach (string message in logMessages)
            {
                EditorGUILayout.HelpBox(message, MessageType.None);
            }
            
            EditorGUILayout.EndScrollView();
            
            if (GUILayout.Button("Clear Log"))
            {
                logMessages.Clear();
            }
        }
        
        private void CheckForFFmpeg()
        {
            try
            {
                System.Diagnostics.Process process = new System.Diagnostics.Process();
                process.StartInfo.FileName = ffmpegPath;
                process.StartInfo.Arguments = "-version";
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.CreateNoWindow = true;
                
                process.Start();
                string output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();
                
                ffmpegFound = output.Contains("ffmpeg version");
                
                if (ffmpegFound)
                {
                    Log("FFmpeg found: " + output.Split('\n')[0]);
                }
                else
                {
                    Log("FFmpeg not found or not working properly");
                }
            }
            catch (System.Exception e)
            {
                ffmpegFound = false;
                Log("Error checking FFmpeg: " + e.Message);
            }
        }
        
        private async void StartImport()
        {
            if (isImporting)
                return;
            
            if (string.IsNullOrEmpty(sourceDatabasePath) || !File.Exists(sourceDatabasePath))
            {
                Log("Error: Asset database file not found");
                return;
            }
            
            isImporting = true;
            importProgress = 0f;
            statusMessage = "Starting import...";
            
            Log("Starting import process...");
            Log("Source: " + sourceDatabasePath);
            Log("Destination: " + outputDirectory);
            
            // Create the output directory if it doesn't exist
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }
            
            // Read the source database
            string databaseJson = File.ReadAllText(sourceDatabasePath);
            var database = JsonUtility.FromJson<Dictionary<string, object>>(databaseJson);
            
            if (database == null)
            {
                Log("Error: Failed to parse asset database");
                isImporting = false;
                return;
            }
            
            try
            {
                // Process database (this would be replaced with actual processing logic)
                int totalAssets = database.Count;
                int processedAssets = 0;
                
                // Placeholder for asset processing simulation
                foreach (var entry in database)
                {
                    await Task.Delay(100); // Simulate processing time
                    
                    processedAssets++;
                    importProgress = (float)processedAssets / totalAssets;
                    statusMessage = $"Processing {processedAssets} of {totalAssets} assets...";
                    
                    // Force UI update
                    Repaint();
                }
                
                Log("Import completed successfully");
                statusMessage = "Import completed";
            }
            catch (System.Exception e)
            {
                Log("Error during import: " + e.Message);
                statusMessage = "Import failed";
            }
            finally
            {
                isImporting = false;
                importProgress = 0f;
                Repaint();
            }
        }
        
        private void Log(string message)
        {
            logMessages.Add(message);
            Debug.Log("[AudioCraft Import] " + message);
            Repaint();
        }
    }
}
```

## Integration Best Practices

Based on our implementations, here are best practices for integrating AudioCraft assets with Unity:

1. **Asset Preparation**
   - Always normalize audio levels for consistent volume
   - Convert to Unity-compatible formats (WAV with 44.1kHz sample rate)
   - Maintain consistent naming conventions
   - Preserve metadata for runtime selection

2. **Organization in Unity**
   - Create a structured folder hierarchy mirroring your asset categories
   - Use audio mixer groups for different asset types
   - Assign appropriate 2D/3D settings based on asset type
   - Set up appropriate audio settings (compression, load type) based on asset usage

3. **Runtime Audio Management**
   - Implement pooling for efficient audio source usage
   - Use metadata-driven sound selection for flexibility
   - Create tag-based lookup systems for runtime filtering
   - Implement appropriate transitions between audio states

4. **Performance Considerations**
   - Use appropriate compression settings for different asset types
   - Stream longer audio files instead of loading them entirely
   - Implement distance-based culling for 3D sounds
   - Monitor memory usage for large audio assets

5. **Workflow Integration**
   - Create editor tools to streamline the import process
   - Generate code constants for type-safe audio references
   - Document asset organization for team members
   - Implement version control strategies for audio assets

## Hands-On Challenge: Create a Dynamic Environment System

**Challenge:** Build a complete dynamic audio environment system for Unity that responds to game state changes.

1. Create an Audio Manager prefab with mixer configurations
2. Implement scene-specific audio zones with overlapping regions
3. Build a parametric audio system that modifies audio based on game variables
4. Create audio transitions that respond to time of day or weather changes
5. Implement an interactive "audio debugging" mode for testing and tuning

Steps to implement:

1. Create the Audio Manager GameObject with proper component configuration
2. Implement AudioTriggerZone components for spatial audio regions
3. Build a ParametricAudioController that modifies audio based on inputs
4. Create transition presets for different environment states
5. Implement audio visualization tools for debugging

## Next Steps

In our next chapter, we'll explore creating interactive audio systems that respond dynamically to player actions and environmental changes. We'll build on our Unity integration to create advanced gameplay-driven audio experiences.

Copyright © 2025 Scott Friedman. Licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).