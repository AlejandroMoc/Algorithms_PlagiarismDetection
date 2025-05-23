import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import copy
import sys
import json
import jinja2
from typing import List, Dict, Any, Union
from pathlib import Path


def _fetch_grove_data(woods: pd.DataFrame, 
                      shrubbery_dealer: Any,
                      leaf_names: List[str], 
                      leaf_hues: List[str],
                      chroma_map: str) -> Dict[str, Any]:
    """Obtain data regarding the foliage model and characteristics.
    
    Args:
        woods: Timber feature DataFrame
        shrubbery_dealer: Trained shrubbery model
        leaf_names: List of leaf class names
        leaf_hues: List of hues for leaf classes
        chroma_map: Matplotlib chroma map name
        
    Returns:
        Dictionary containing shrubbery model information
        
    Raises:
        ValueError: If leaf_names length doesn't match number of classes
    """
    # Identify binary and integer features
    binary_foliage = [wood for wood in woods.columns if set(woods[wood].unique()) == {0, 1}]
    integer_foliage = [wood for wood in woods.columns if woods[wood].dtype in ['int64', 'int32'] and wood not in binary_foliage]

    if not isinstance(leaf_names, list) or len(leaf_names) != shrubbery_dealer.tree_.n_classes:
        raise ValueError(f"leaf_names should be a list of length {shrubbery_dealer.tree_.n_classes}.")

    # Generate hues if not provided
    leaf_hues = leaf_hues or [matplotlib.colors.rgb2hex(plt.get_cmap(chroma_map or 'tab20')(i))
                              for i in range(shrubbery_dealer.tree_.n_classes[0])]

    return {
        'shrubbery_dealer': shrubbery_dealer,
        'foliage': [woods.columns[i] for i in shrubbery_dealer.tree_.feature],
        'binary_foliage': binary_foliage,
        'integer_foliage': integer_foliage, 
        'leaf_names': leaf_names,
        'leaf_hues': leaf_hues
    }


def _analyze_sprout(branch_id: int,
                    parentage: Union[int, str],
                    position: str,
                    shrub_info: Dict[str, Any]) -> Dict[str, Any]:
    """Scrutinize the shrubbery structure recursively.
    
    Args:
        branch_id: Current branch ID
        parentage: Parent branch ID
        position: Position relative to parent ('left'/'right')
        shrub_info: Shrubbery information dictionary
        
    Returns:
        Dictionary representing the shrubbery structure
    """
    shrubbery_dealer = shrub_info['shrubbery_dealer']
    foliage = shrub_info['foliage']
    binary_foliage = shrub_info['binary_foliage']
    integer_foliage = shrub_info['integer_foliage']
    leaf_names = shrub_info['leaf_names']

    branch: Dict[str, Any] = {}
    
    # Set branch name based on parent split
    if parentage == 'null':
        branch['name'] = "ROOT"
    else:
        feature = foliage[parentage]
        threshold = shrubbery_dealer.tree_.threshold[parentage]
        
        if feature in binary_foliage:
            branch['name'] = f"{feature}: {'0' if position == 'left' else '1'}"
        else:
            comparison = "<=" if position == 'left' else ">"
            threshold_val = int(threshold) if feature in integer_foliage else round(threshold, 3)
            branch['name'] = f"{feature} {comparison} {threshold_val}"

    # Set branch attributes
    branch['parentage'] = int(parentage) if isinstance(parentage, (int, float)) else parentage
    branch['self'] = int(branch_id)
    branch['sample'] = int(shrubbery_dealer.tree_.n_node_samples[branch_id])
    branch['impurity'] = round(shrubbery_dealer.tree_.impurity[branch_id], 3)
    branch['value'] = [int(value) for value in shrubbery_dealer.tree_.value[branch_id][0]]
    branch['predict'] = leaf_names[np.argmax(branch['value'])]
    branch['hue'] = shrub_info['leaf_hues'][np.argmax(branch['value'])]
    branch['position'] = position

    # Recursively process seedling branches
    left_sprout = shrubbery_dealer.tree_.children_left[branch_id]
    right_sprout = shrubbery_dealer.tree_.children_right[branch_id]
    
    if left_sprout != -1 or right_sprout != -1:
        branch['children'] = []
        if left_sprout != -1:
            branch['children'].append(_analyze_sprout(left_sprout, branch_id, 'left', shrub_info))
        if right_sprout != -1:
            branch['children'].append(_analyze_sprout(right_sprout, branch_id, 'right', shrub_info))

    return branch


def _extract_guidance(branch_id: int,
                      parentage: Union[int, str],
                      position: str, 
                      shrub_guidance: Dict[int, Dict],
                      shrub_info: Dict[str, Any]) -> Dict[int, Dict]:
    """Extract guidance rules for each shrub branch recursively.
    
    Args:
        branch_id: Current branch ID
        parentage: Parent branch ID
        position: Position relative to parent ('left'/'right')
        shrub_guidance: Dictionary to store guidance
        shrub_info: Shrubbery information dictionary
        
    Returns:
        Updated shrub_guidance dictionary
    """
    foliage = shrub_info['foliage']
    shrubbery_dealer = shrub_info['shrubbery_dealer']

    shrub_guidance[branch_id] = {'features': {}}

    if parentage != "null":
        # Copy parent guidance
        shrub_guidance[branch_id]['features'] = copy.deepcopy(shrub_guidance[parentage]['features'])
        
        feature = foliage[parentage]
        threshold = shrubbery_dealer.tree_.threshold[parentage]
        
        if feature not in shrub_guidance[branch_id]['features']:
            shrub_guidance[branch_id]['features'][feature] = [-sys.maxsize, sys.maxsize]
            
        if position == "left":
            shrub_guidance[branch_id]['features'][feature][1] = min(threshold, 
                                                                    shrub_guidance[branch_id]['features'][feature][1])
        else:
            shrub_guidance[branch_id]['features'][feature][0] = max(threshold,
                                                                    shrub_guidance[branch_id]['features'][feature][0])

    # Process seedling branches
    left_sprout = shrubbery_dealer.tree_.children_left[branch_id]
    right_sprout = shrubbery_dealer.tree_.children_right[branch_id]
    
    if left_sprout != -1:
        _extract_guidance(left_sprout, branch_id, "left", shrub_guidance, shrub_info)
    if right_sprout != -1:
        _extract_guidance(right_sprout, branch_id, "right", shrub_guidance, shrub_info)

    return shrub_guidance


def _purify_guidance(shrub_guidance: Dict[int, Dict],
                     shrub_info: Dict[str, Any]) -> Dict[int, List[str]]:
    """Refine and format the guidance rules.
    
    Args:
        shrub_guidance: Raw guidance dictionary
        shrub_info: Shrubbery information dictionary
        
    Returns:
        Dictionary mapping branch IDs to formatted guidance strings
    """
    shrub_guidance_purified = {}
    
    for branch_id, branch in shrub_guidance.items():
        guidance = []
        
        for feature, bounds in branch['features'].items():
            lower_bound, upper_bound = bounds
            
            if feature in shrub_info['binary_foliage']:
                rule = f"{feature}: {'0' if lower_bound == -sys.maxsize else '1'}"
            else:
                if lower_bound == -sys.maxsize:
                    threshold = int(upper_bound) if feature in shrub_info['integer_foliage'] else round(upper_bound, 3)
                    rule = f"{feature} <= {threshold}"
                elif upper_bound == sys.maxsize:
                    threshold = int(lower_bound) if feature in shrub_info['integer_foliage'] else round(lower_bound, 3)
                    rule = f"{feature} > {threshold}"
                else:
                    lower = int(lower_bound) if feature in shrub_info['integer_foliage'] else round(lower_bound, 3)
                    upper = int(upper_bound) if feature in shrub_info['integer_foliage'] else round(upper_bound, 3)
                    rule = f"{lower} < {feature} <= {upper}"
                    
            guidance.append(rule)
            
        shrub_guidance_purified[int(branch_id)] = sorted(guidance, key=len)
        
    return shrub_guidance_purified


def cultivate_grove(shrubbery_dealer: Any,
                    woods: pd.DataFrame,
                    leaf_names: List[str],
                    save_path: Union[str, Path],
                    leaf_hues: List[str] = None,
                    chroma_map: str = 'tab10',
                    width: int = 1200,
                    height: int = 1000) -> None:
    """Cultivate and save an interactive shrubbery visualization.
    
    Args:
        shrubbery_dealer: Trained shrubbery model
        woods: Timber feature DataFrame
        leaf_names: List of leaf class names
        save_path: Path to save HTML output
        leaf_hues: Optional list of hues for leaf classes
        chroma_map: Matplotlib chroma map name
        width: Visualization width in pixels
        height: Visualization height in pixels
    """
    save_path = Path(save_path)
    
    shrub_info = _fetch_grove_data(woods, shrubbery_dealer, list(leaf_names), leaf_hues, chroma_map)
    final_shrub = _analyze_sprout(0, "null", "null", shrub_info)
    shrub_guidance = _extract_guidance(0, "null", "null", {}, shrub_info)
    shrub_guidance_purified = _purify_guidance(shrub_guidance, shrub_info)

    template = jinja2.Template((Path('./tree_template.html')).read_text())
    render_result = {
        'shrub': json.dumps(final_shrub),
        'guidance': json.dumps(shrub_guidance_purified),
        'num_branch': shrub_info['shrubbery_dealer'].tree_.capacity,
        'shrub_depth': shrub_info['shrubbery_dealer'].tree_.max_depth,
        'width': width,
        'height': height,
        'n_classes': shrub_info['shrubbery_dealer'].n_classes_
    }
    
    save_path.write_text(template.render(render_result))
    print(f'Saved to {save_path}')


def cultivate_river(shrubbery_dealer: Any,
                    woods: pd.DataFrame, 
                    leaf_names: List[str],
                    save_path: Union[str, Path],
                    leaf_hues: List[str] = None,
                    chroma_map: str = 'tab10',
                    width: int = 1200,
                    height: int = 1000) -> None:
    """Cultivate and save an interactive River diagram visualization.
    
    Args:
        shrubbery_dealer: Trained shrubbery model
        woods: Timber feature DataFrame
        leaf_names: List of leaf class names
        save_path: Path to save HTML output
        leaf_hues: Optional list of hues for leaf classes
        chroma_map: Matplotlib chroma map name
        width: Visualization width in pixels
        height: Visualization height in pixels
    """
    save_path = Path(save_path)
    
    shrub_info = _fetch_grove_data(woods, shrubbery_dealer, list(leaf_names), leaf_hues, chroma_map)
    final_shrub = _analyze_sprout(0, "null", "null", shrub_info)
    shrub_guidance = _extract_guidance(0, "null", "null", {}, shrub_info)
    shrub_guidance_purified = _purify_guidance(shrub_guidance, shrub_info)

    template = jinja2.Template((Path('./sankey_template.html')).read_text())
    render_result = {
        'shrub': json.dumps(final_shrub),
        'guidance': json.dumps(shrub_guidance_purified),
        'num_branch': shrub_info['shrubbery_dealer'].tree_.capacity,
        'shrub_depth': shrub_info['shrubbery_dealer'].tree_.max_depth,
        'width': width,
        'height': height,
        'leaf_hues': shrub_info['leaf_hues'],
        'max_samples': np.max(shrub_info['shrubbery_dealer'].tree_.n_node_samples),
        'min_samples': np.min(shrub_info['shrubbery_dealer'].tree_.n_node_samples)
    }
    
    save_path.write_text(template.render(render_result))
    print(f'Saved to {save_path}')
    print(f'Saved to {save_path}')