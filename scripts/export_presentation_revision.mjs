// Exact Beamer image rendition through the supplied Artifact Tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const runtime='/Users/xiao/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const skill='/Users/xiao/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const folder=path.join(root,'presentations/math_tokenizer');
const prep=JSON.parse(await fs.readFile(path.join(folder,'build/export_revision_input.json'),'utf8'));
const presentation=Presentation.create({slideSize:{width:960,height:720}});
for(let i=0;i<prep.images.length;i++){
  const slide=presentation.slides.add();
  slide.images.add({blob:new Uint8Array(await fs.readFile(prep.images[i])),contentType:'image/png',
    alt:`Beamer presentation page ${i+1}`,fit:'contain',position:{left:0,top:0,width:960,height:720}});
  slide.speakerNotes.text=prep.notes[i]+'\n\nEditable master: mathematical_tokenizer.tex in companion source bundle; this page preserves the Beamer rendering.';
}
const staging=path.join(folder,'build',`artifact_export_${Date.now()}`);await fs.mkdir(staging,{recursive:true});
const candidate=path.join(staging,'candidate.pptx');await(await PresentationFile.exportPptx(presentation)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const final=prep.finalPptx ?? path.join(folder,'output/Design_of_a_Specialized_Tokenizer_for_Mathematical_Text_Updated.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,
  pythonExecutable:path.join(runtime,'python/bin/python3'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:prep.images.length,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  verifyArtifactToolImport:true,receiptPath:path.join(staging,'validation.json')});
await fs.writeFile(path.join(folder,'build/artifact_export/result.json'),JSON.stringify(result,null,2));
console.log('Exported revised Beamer PowerPoint',prep.images.length,'slides',final);
