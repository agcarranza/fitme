# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TF-Slim grouped API. Please see README.md for details and usage."""
# pylint: disable=unused-import

# Collapse tf-slim into a single namespace.
from slim import inception_v1
from slim import inception_v3
from slim import losses
from slim import ops
from slim import scopes
from slim import variables
from slim.scopes import arg_scope
