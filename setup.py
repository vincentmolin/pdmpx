# Copyright 2023 Vincent Molin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import setuptools


base_requires = ["jax", "numba"]
tests_requires = [
    "jaxlib",
    "pytest",
]

setuptools.setup(
    name="pdmpx",
    description="PDMPs in JAX",
    version="0.0.1",
    license="Apache 2.0",
    author="Vincent Molin",
    author_email="molinv@chalmers.se",
    install_requires=base_requires,
    extras_require={
        "tests": tests_requires,
    },
    url="https://github.com/vincentmolin/pdmpx",
    packages=setuptools.find_packages(),  # "]),
    python_requires=">=3",
)
