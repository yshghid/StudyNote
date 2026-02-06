export interface FileEntry {
  name: string;
  type: 'directory' | 'file' | 'page';
  path: string;
  title?: string;  // title from _index.md frontmatter
  children?: FileEntry[];
}

export interface NotebookCell {
  cell_type: 'code' | 'markdown' | 'raw';
  source: string[];
  outputs?: CellOutput[];
  execution_count?: number | null;
}

export interface CellOutput {
  output_type: 'stream' | 'execute_result' | 'display_data' | 'error';
  text?: string[];
  data?: Record<string, string[]>;
  name?: string;
  ename?: string;
  evalue?: string;
  traceback?: string[];
}

export interface Notebook {
  cells: NotebookCell[];
  metadata: Record<string, unknown>;
  nbformat: number;
  nbformat_minor: number;
}

export interface ExecutionResult {
  success: boolean;
  output: string;
  error?: string;
}
