import axios from 'axios';
import type { FileEntry, Notebook, ExecutionResult } from '../types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export async function getDocsStructure(): Promise<FileEntry[]> {
  const response = await api.get<FileEntry[]>('/api/docs/structure');
  return response.data;
}

export async function getIndexContent(): Promise<string> {
  const response = await api.get<{ content: string }>('/api/index');
  return response.data.content;
}

export async function getPageContent(path: string): Promise<{ content: string; basePath: string }> {
  const response = await api.get<{ content: string; basePath: string }>('/api/page', {
    params: { path },
  });
  return response.data;
}

export async function getNotebook(path: string): Promise<Notebook> {
  const response = await api.get<Notebook>('/api/notebook', {
    params: { path },
  });
  return response.data;
}

export async function executeCell(
  notebookName: string,
  code: string,
  cellIndex: number
): Promise<ExecutionResult> {
  const response = await api.post<ExecutionResult>('/api/execute', {
    notebook_name: notebookName,
    code,
    cell_index: cellIndex,
  });
  return response.data;
}
