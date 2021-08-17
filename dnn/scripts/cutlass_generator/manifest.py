#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import enum
import os.path
import shutil

from library import *
from gemm_operation import *
from conv2d_operation import *  

###################################################################################################

class EmitOperationKindLibrary:
  def __init__(self, generated_path, kind, args):
    self.generated_path = generated_path
    self.kind = kind
    self.args = args

    self.emitters = {
      OperationKind.Gemm: EmitGemmConfigurationLibrary
      , OperationKind.Conv2d: EmitConv2dConfigurationLibrary
    }

    self.configurations = [];

    self.header_template ="""
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

"""
    self.entry_template = """

//
// Entry point to construct operations
//
void initialize_all_${operation_name}_operations(Manifest &manifest) {
"""
    self.configuration_prototype_template = "void initialize_${configuration_name}(Manifest &manifest);\n"
    self.configuration_template ="  initialize_${configuration_name}(manifest);\n"

    self.epilogue_template ="""

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

"""

  #
  def __enter__(self):
    self.operation_path = os.path.join(self.generated_path, OperationKindNames[self.kind])
    os.mkdir(self.operation_path)

    self.top_level_path = os.path.join(self.operation_path, "all_%s_operations.cu" % OperationKindNames[self.kind])

    self.top_level_file = open(self.top_level_path, "w")
    self.top_level_file.write(self.header_template)

    self.source_files = [self.top_level_path,]

    return self

  #
  def emit(self, configuration_name, operations):

    with self.emitters[self.kind](self.operation_path, configuration_name) as configuration_emitter:
      for operation in operations:
        configuration_emitter.emit(operation)
      
      self.source_files.append(configuration_emitter.configuration_path)

    self.configurations.append(configuration_name)
    self.top_level_file.write(SubstituteTemplate(self.configuration_prototype_template, {'configuration_name': configuration_name} ))

  #
  def __exit__(self, exception_type, exception_value, traceback):
    self.top_level_file.write(SubstituteTemplate(self.entry_template, {'operation_name': OperationKindNames[self.kind]}))

    for configuration_name in self.configurations:
      self.top_level_file.write(SubstituteTemplate(self.configuration_template, {'configuration_name': configuration_name}))

    self.top_level_file.write(self.epilogue_template)
    self.top_level_file.close()

###################################################################################################
###################################################################################################

class Options:
  def __init__(self):
    pass

###################################################################################################

#
class Manifest:

  #
  def __init__(self, args):
    self.operations = {}
    self.args = args

    architectures = args.architectures.split(';') if len(args.architectures) else ['50',]
    self.compute_capabilities = [int(x) for x in architectures]
    
    self.selected_kernels = []
    
    if args.operations == 'all':
      self.operations_enabled = []
    else:

      operations_list = [
        OperationKind.Gemm
        , OperationKind.Conv2d     
      ] 

      self.operations_enabled = [x for x in operations_list if OperationKindNames[x] in args.operations.split(',')]

    if args.kernels == 'all':
      self.kernel_names = []
    else:
      self.kernel_names = [x for x in args.kernels.split(',') if x != '']

    self.ignore_kernel_names = [x for x in args.ignore_kernels.split(',') if x != '']

    if args.kernel_filter_file is None:
        self.kernel_filter_list = []
    else:
        self.kernel_filter_list = self.get_kernel_filters(args.kernel_filter_file)


    self.operation_count = 0
    self.operations_by_name = {}
    self.top_level_prologue = '''

#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

${prototypes}

void initialize_all(Manifest &manifest) {

'''
    self.top_level_reserve = '  manifest.reserve(${operation_count});\n\n'
    self.top_level_epilogue = '''
}

} // namespace library
} // namespace cutlass

'''


  def get_kernel_filters (self, kernelListFile):
    if os.path.isfile(kernelListFile):
        with open(kernelListFile, 'r') as fileReader:
            lines = [line.rstrip() for line in fileReader if not line.startswith("#")]
        
        lines = [re.compile(line) for line in lines if line]
        return lines
    else:
        return []



  def filter_out_kernels(self, kernel_name, kernel_filter_list):

    for kernel_filter_re in kernel_filter_list:
        if kernel_filter_re.search(kernel_name) is not None:
            return True
        
    return False

    
  #
  def _filter_string_matches(self, filter_string, haystack):
    ''' Returns true if all substrings appear in the haystack in order'''
    substrings = filter_string.split('*')
    for sub in substrings:
      idx = haystack.find(sub)
      if idx < 0:
        return False
      haystack = haystack[idx + len(sub):]
    return True

  #
  def filter(self, operation):
    ''' Filtering operations based on various criteria'''

    # filter based on compute capability
    enabled = False
    for cc in self.compute_capabilities:
      if cc >= operation.tile_description.minimum_compute_capability and \
        cc <= operation.tile_description.maximum_compute_capability:

        enabled = True
        break

    if not enabled:
      return False

    if len(self.operations_enabled) and not operation.operation_kind in self.operations_enabled:
      return False

    # eliminate duplicates
    if operation.procedural_name() in self.operations_by_name.keys():
      return False

    # Filter based on list of valid substrings
    if len(self.kernel_names):
      name = operation.procedural_name()
      enabled = False

      # compare against the include list
      for name_substr in self.kernel_names:
        if self._filter_string_matches(name_substr, name):
          enabled = True
          break

      # compare against the exclude list
      for name_substr in self.ignore_kernel_names:
        if self._filter_string_matches(name_substr, name):
          enabled = False
          break
          
    if len(self.kernel_filter_list) > 0:
        enabled = False
        if self.filter_out_kernels(operation.procedural_name(), self.kernel_filter_list):
            enabled = True


    # todo: filter based on compute data type
    return enabled
  #

  #
  def append(self, operation):
    ''' 
      Inserts the operation.

      operation_kind -> configuration_name -> []
    '''

    if self.filter(operation):
    
      self.selected_kernels.append(operation.procedural_name())

      self.operations_by_name[operation.procedural_name()] = operation

      # add the configuration
      configuration_name = operation.configuration_name()

      if operation.operation_kind not in self.operations.keys():
        self.operations[operation.operation_kind] = {}

      if configuration_name not in self.operations[operation.operation_kind].keys():
        self.operations[operation.operation_kind][configuration_name] = []

      self.operations[operation.operation_kind][configuration_name].append(operation)
      self.operation_count += 1
  #

  #
  def emit(self, target = GeneratorTarget.Library):

    operation_emitters = {
      GeneratorTarget.Library: EmitOperationKindLibrary 
    }

    generated_path = os.path.join(self.args.curr_build_dir, 'generated')

    # create generated/
    if os.path.exists(generated_path):
      shutil.rmtree(generated_path)

    os.mkdir(generated_path)

    source_files = []

    top_level_path = os.path.join(generated_path, 'initialize_all.cpp')
    with open(top_level_path, 'w') as top_level_file:

      if target == GeneratorTarget.Library:
        source_files.append(top_level_path)

      prototypes = []
      for operation_kind, configurations in self.operations.items():
        prototypes.append(SubstituteTemplate(
          "void initialize_all_${operation_kind}_operations(Manifest &manifest);",
          {'operation_kind': OperationKindNames[operation_kind]}))

      top_level_file.write(SubstituteTemplate(self.top_level_prologue,
        {'prototypes': "\n".join(prototypes)}))

      top_level_file.write(SubstituteTemplate(
        self.top_level_reserve, {'operation_count': str(self.operation_count)}))

      # for each operation kind, emit initializer for all configurations
      for operation_kind, configurations in self.operations.items():
        
        with operation_emitters[target](generated_path, operation_kind, self.args) as operation_kind_emitter:
          for configuration_name, operations in configurations.items():
            operation_kind_emitter.emit(configuration_name, operations)

          source_files += operation_kind_emitter.source_files

        top_level_file.write(SubstituteTemplate(
          "  initialize_all_${operation_kind}_operations(manifest);\n",
          {'operation_kind': OperationKindNames[operation_kind]}))

      top_level_file.write(self.top_level_epilogue)

    # write the manifest.cmake file containing paths from all targets
    manifest_path = os.path.join(generated_path, "manifest.cmake")
    with open(manifest_path, "w") as manifest_file:

      target_name = 'cutlass_library_objs'

      target_text = SubstituteTemplate("""cutlass_target_sources(
  ${target_name}
  BATCH_SOURCES ON
  PRIVATE
""", { 'target_name': target_name})

      manifest_file.write(target_text)

      for source_file in source_files:
        manifest_file.write("    %s\n" % str(source_file.replace('\\', '/')))
      manifest_file.write(")")
  #

###################################################################################################

def GenerateManifest(args, operations, output_dir):
  manifest_path = os.path.join(output_dir, "all_%s_%s_operations.cu" % (args.operations, args.type))
  f = open(manifest_path, "w")
  f.write("""
/*
 Generated by generator.py - Do not edit.
*/

#if __CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ >= 2)

#include "cutlass/cutlass.h"
#include "src/cuda/cutlass/library.h"
#include "src/cuda/cutlass/manifest.h"

namespace cutlass {
namespace library {

""")
  for op in operations:
      f.write("void initialize_%s(Manifest &manifest);\n" % op.procedural_name())

  f.write("""
void initialize_all_%s_%s_operations(Manifest &manifest) {
""" % (args.operations, args.type))

  for op in operations:
    f.write("    initialize_%s(manifest);\n" % op.procedural_name())

  f.write("""
}

}  // namespace library
}  // namespace cutlass

#endif
""")
  f.close()
