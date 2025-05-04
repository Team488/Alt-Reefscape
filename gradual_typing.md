# Gradual Typing Migration Strategy

This document outlines the strategy for gradually migrating the codebase to use proper type hints.

## Current Progress

The following files have been fully type-annotated:

- `src/pathplanning/PathFind.py`
- `src/pathplanning/AStar.py`
- `src/abstract/Agent.py`
- `src/abstract/Order.py`
- `src/abstract/Capture.py`
- `src/abstract/inferencerBackend.py`
- `src/abstract/depthCamera.py`
- `src/abstract/BaseDemo.py`
- `src/Core/Agents/Abstract/CameraUsingAgentBase.py`
- `src/Core/Agents/Abstract/InferenceAgent.py`
- `src/Core/Agents/Abstract/TimestampRegulatedAgentBase.py`
- `src/Core/Agents/Abstract/PositionLocalizingAgentBase.py`
- `src/Core/Agents/Abstract/ObjectLocalizingAgentBase.py`
- `src/Core/Agents/Abstract/ReefTrackingAgentBase.py`

### Implementation Classes

- `src/Core/Agents/AgentExample.py`
- `src/Core/Agents/FrameDisplayer.py`
- `src/Core/Agents/ReefAndObjectLocalizer.py`
- `src/Core/Agents/VideoWriterAgent.py`

### Core Operators

- `src/Core/PropertyOperator.py`
- `src/Core/ConfigOperator.py`
- `src/Core/UpdateOperator.py`
- `src/Core/LogManager.py`
- `src/Core/TimeOperator.py`

### Tools

- `src/tools/NtUtils.py`
- `src/tools/XTutils.py`
- `src/tools/UnitConversion.py`
- `src/tools/Calculator.py`
- `src/tools/CameraUtils.py`
- `src/tools/CameraTransform.py`
- `src/tools/positionTranslations.py`
- `src/tools/Data.py`
- `src/tools/CsvParser.py`

## Ignored Files

The following files have particularly complex typing issues and are ignored for now:

- `src/reefTracking/ReefVisualizer.py`
- `src/reefTracking/SimulateReefVisualzer.py`
- `src/tools/histogramCreator.py`
- `src/mapinternals/probmap.py`

## Next Steps

1. Add type annotations to one file at a time, prioritizing:
   - Files with fewer dependencies
   - Core functionality files
   - Files that are actively being worked on

2. For each file:
   - Add imports from typing (List, Dict, Tuple, Optional, etc.)
   - Add type annotations to function parameters
   - Add return type annotations
   - Add type annotations to class variables
   - Add type annotations to local variables where necessary

3. After each file is annotated:
   - Run `mypy` on that specific file to check for errors
   - Fix any type errors
   - Add the file to the mypy.ini configuration to enforce strict checking

4. Gradually increase the strictness of mypy checking:
   - Enable `disallow_incomplete_defs` for more files
   - Enable `disallow_untyped_defs` for more files
   - Enable `check_untyped_defs` for more files

## Common Issues

- Missing imports from typing
- Union types needed for variables that can be multiple types
- None checks for variables that might be None
- Type errors in mathematical operations
- Missing type annotations for function parameters and return values

## Best Practices

- Use `Optional[Type]` for variables that might be None
- Use `Union[Type1, Type2]` for variables that can be multiple types
- Use `Any` sparingly, only when the type is truly dynamic
- Add type annotations to all function parameters and return values
- Add type annotations to class variables in __init__ methods
- Add type annotations to local variables only when necessary for clarity
