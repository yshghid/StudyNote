import React, { useState, useEffect, useRef } from 'react';
import { Play, Loader2 } from 'lucide-react';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import javascript from 'highlight.js/lib/languages/javascript';
import typescript from 'highlight.js/lib/languages/typescript';
import bash from 'highlight.js/lib/languages/bash';
import json from 'highlight.js/lib/languages/json';
import 'highlight.js/styles/github.css';
import type { Notebook, NotebookCell, ExecutionResult } from '../types';
import { executeCell } from '../api';
import { MarkdownViewer } from './MarkdownViewer';

// Register languages
hljs.registerLanguage('python', python);
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('json', json);

// Get language from notebook metadata
function getLanguage(notebook: Notebook): string {
  const metadata = notebook.metadata as {
    kernelspec?: { language?: string };
    language_info?: { name?: string };
  };
  const lang = metadata?.kernelspec?.language ||
               metadata?.language_info?.name ||
               'python';
  return String(lang).toLowerCase();
}

// Code block with syntax highlighting
function CodeBlock({ code, language }: { code: string; language: string }) {
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (codeRef.current) {
      hljs.highlightElement(codeRef.current);
    }
  }, [code, language]);

  return (
    <pre className="flex-1 overflow-x-auto">
      <code ref={codeRef} className={`language-${language}`}>
        {code}
      </code>
    </pre>
  );
}

interface NotebookViewerProps {
  notebook: Notebook;
  notebookName: string;
}

interface CellState {
  isRunning: boolean;
  output: string | null;
  error: string | null;
}

export function NotebookViewer({ notebook, notebookName }: NotebookViewerProps) {
  const [cellStates, setCellStates] = useState<Record<number, CellState>>({});

  // Parse output and render images
  const renderOutputWithImages = (output: string) => {
    const imageRegex = /<!--IMAGE:(data:image\/[^;]+;base64,[^:]+):IMAGE-->/g;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;
    let imageIndex = 0;

    while ((match = imageRegex.exec(output)) !== null) {
      // Add text before the image
      if (match.index > lastIndex) {
        parts.push(output.slice(lastIndex, match.index));
      }
      // Add the image
      parts.push(
        <img
          key={`img-${imageIndex++}`}
          src={match[1]}
          alt="Output"
          className="max-w-full my-2 rounded border border-gray-200"
        />
      );
      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < output.length) {
      parts.push(output.slice(lastIndex));
    }

    return parts.length > 0 ? parts : output;
  };

  const handleRunCell = async (cell: NotebookCell, index: number) => {
    if (cell.cell_type !== 'code') return;

    const code = Array.isArray(cell.source) ? cell.source.join('') : cell.source;

    setCellStates((prev) => ({
      ...prev,
      [index]: { isRunning: true, output: null, error: null },
    }));

    try {
      const result: ExecutionResult = await executeCell(notebookName, code, index);
      setCellStates((prev) => ({
        ...prev,
        [index]: {
          isRunning: false,
          output: result.success ? result.output : null,
          error: result.success ? null : result.error || 'Execution failed',
        },
      }));
    } catch (err) {
      setCellStates((prev) => ({
        ...prev,
        [index]: {
          isRunning: false,
          output: null,
          error: err instanceof Error ? err.message : 'Failed to execute cell',
        },
      }));
    }
  };

  const renderCellOutput = (cell: NotebookCell, index: number) => {
    const state = cellStates[index];

    // Show execution result from state if available
    if (state) {
      if (state.isRunning) {
        return (
          <div className="cell-output text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
            Running...
          </div>
        );
      }
      if (state.error) {
        return <div className="cell-output error">{state.error}</div>;
      }
      if (state.output) {
        return <div className="cell-output">{renderOutputWithImages(state.output)}</div>;
      }
    }

    // Show original notebook outputs
    if (!cell.outputs || cell.outputs.length === 0) return null;

    return (
      <div className="cell-output">
        {cell.outputs.map((output, outputIndex) => {
          if (output.output_type === 'stream' && output.text) {
            return <div key={outputIndex}>{output.text.join('')}</div>;
          }
          if (output.output_type === 'execute_result' && output.data) {
            const textData = output.data['text/plain'];
            if (textData) {
              return <div key={outputIndex}>{textData.join('')}</div>;
            }
          }
          if (output.output_type === 'error') {
            return (
              <div key={outputIndex} className="text-red-600">
                {output.ename}: {output.evalue}
                {output.traceback && (
                  <pre className="mt-2 text-xs">{output.traceback.join('\n')}</pre>
                )}
              </div>
            );
          }
          return null;
        })}
      </div>
    );
  };

  const renderCell = (cell: NotebookCell, index: number) => {
    const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
    const state = cellStates[index];

    if (cell.cell_type === 'markdown') {
      return (
        <div key={index} className="mb-4">
          <MarkdownViewer content={source} />
        </div>
      );
    }

    if (cell.cell_type === 'code') {
      const language = getLanguage(notebook);
      return (
        <div key={index} className="code-cell">
          <div className="flex items-start">
            <button
              onClick={() => handleRunCell(cell, index)}
              disabled={state?.isRunning}
              className="flex-shrink-0 p-1 m-1 rounded hover:bg-gray-200 transition-colors disabled:opacity-50"
              title="Run cell"
            >
              {state?.isRunning ? (
                <Loader2 className="w-3 h-3 animate-spin text-gray-500" />
              ) : (
                <Play className="w-3 h-3 text-gray-500" />
              )}
            </button>
            <CodeBlock code={source} language={language} />
          </div>
          {renderCellOutput(cell, index)}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="max-w-4xl">
      {notebook.cells.map((cell, index) => renderCell(cell, index))}
    </div>
  );
}
