import { Home } from 'lucide-react';
import type { FileEntry } from '../types';

interface SidebarProps {
  structure: FileEntry[];
  selectedFile: string | null;
  onSelectFile: (path: string) => void;
  onGoHome: () => void;
}

export function Sidebar({ structure, selectedFile, onSelectFile, onGoHome }: SidebarProps) {
  const renderEntry = (entry: FileEntry, depth: number = 0) => {
    const isSelected = selectedFile === entry.path;

    if (entry.type === 'directory') {
      return (
        <div key={entry.path} style={{ marginLeft: depth * 16 }} className={depth === 0 ? 'mt-8' : ''}>
          <div className="font-bold text-gray-800 py-1">
            {entry.name}
          </div>
          {entry.children?.map((child) => renderEntry(child, depth + 1))}
        </div>
      );
    }

    if (entry.type === 'page') {
      return (
        <div
          key={entry.path}
          className={`py-1 cursor-pointer transition-colors ${
            isSelected ? 'text-gray-400' : 'text-gray-800 hover:text-gray-500'
          }`}
          onClick={() => onSelectFile(entry.path)}
        >
          {entry.title || entry.name}
        </div>
      );
    }

    // File - extract name without extension
    const displayName = entry.name.replace(/\.ipynb$/, '');

    return (
      <div
        key={entry.path}
        style={{ marginLeft: depth * 16 }}
        className={`py-1 cursor-pointer transition-colors ${
          isSelected ? 'text-gray-400' : 'text-gray-800 hover:text-gray-500'
        }`}
        onClick={() => onSelectFile(entry.path)}
      >
        {displayName}
      </div>
    );
  };

  return (
    <aside className="fixed left-40 top-0 w-64 min-h-screen p-6">
      <div
        className="mt-5 mb-6 cursor-pointer hover:text-gray-400 transition-colors"
        onClick={onGoHome}
      >
        <Home className="w-5 h-5 text-gray-600" />
      </div>
      <nav className="text-sm mt-12">
        {structure.map((entry) => renderEntry(entry))}
      </nav>
    </aside>
  );
}
