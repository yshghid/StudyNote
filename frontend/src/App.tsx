import { useEffect, useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import type { FileEntry, Notebook } from './types';
import { getDocsStructure, getIndexContent, getNotebook, getPageContent } from './api';
import { Sidebar } from './components/Sidebar';
import { MarkdownViewer } from './components/MarkdownViewer';
import { NotebookViewer } from './components/NotebookViewer';

type ViewType = 'index' | 'page' | 'notebook';

function App() {
  const navigate = useNavigate();
  const location = useLocation();

  const [structure, setStructure] = useState<FileEntry[]>([]);
  const [indexContent, setIndexContent] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [viewType, setViewType] = useState<ViewType>('index');
  const [pageContent, setPageContent] = useState<string>('');
  const [pageBasePath, setPageBasePath] = useState<string>('');
  const [notebook, setNotebook] = useState<Notebook | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      const [structureData, indexData] = await Promise.all([
        getDocsStructure(),
        getIndexContent(),
      ]);
      setStructure(structureData);
      setIndexContent(indexData);
    } catch (err) {
      setError('Failed to load data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Find entry type from structure
  const findEntryType = useCallback((path: string, entries: FileEntry[]): 'file' | 'page' | null => {
    for (const entry of entries) {
      if (entry.path === path) {
        return entry.type === 'page' ? 'page' : entry.type === 'file' ? 'file' : null;
      }
      if (entry.children) {
        const found = findEntryType(path, entry.children);
        if (found) return found;
      }
    }
    return null;
  }, []);

  // Load content based on path
  const loadContentFromPath = useCallback(async (path: string) => {
    if (structure.length === 0) return;

    const entryType = findEntryType(path, structure);
    if (!entryType) {
      // Try to load as notebook directly (for slug-based links)
      try {
        const notebookData = await getNotebook(path);
        setSelectedFile(path);
        setNotebook(notebookData);
        setPageContent('');
        setViewType('notebook');
        return;
      } catch {
        return;
      }
    }

    setSelectedFile(path);
    setNotebook(null);
    setPageContent('');
    setError(null);

    try {
      if (entryType === 'page') {
        const data = await getPageContent(path);
        setPageContent(data.content);
        setPageBasePath(data.basePath);
        setViewType('page');
      } else {
        const notebookData = await getNotebook(path);
        setNotebook(notebookData);
        setViewType('notebook');
      }
    } catch (err) {
      setError('Failed to load content');
      console.error(err);
    }
  }, [structure, findEntryType]);

  // Decode path segments individually
  const decodePath = (path: string) => {
    return path.split('/').map(segment => decodeURIComponent(segment)).join('/');
  };

  // Handle URL changes
  useEffect(() => {
    const rawPath = location.pathname;
    if (rawPath === '/' || rawPath === '') {
      setSelectedFile(null);
      setNotebook(null);
      setPageContent('');
      setViewType('index');
    } else if (rawPath.startsWith('/')) {
      const filePath = decodePath(rawPath.slice(1));
      loadContentFromPath(filePath);
    }
  }, [location.pathname, loadContentFromPath]);

  // Encode path segments individually (keep slashes)
  const encodePath = (path: string) => {
    return path.split('/').map(segment => encodeURIComponent(segment)).join('/');
  };

  const handleSelectFile = (path: string) => {
    navigate('/' + encodePath(path));
  };

  // Handle link clicks in page markdown
  const handlePageLinkClick = (href: string) => {
    if (href.startsWith('#')) {
      const slug = href.slice(1);
      const slugPath = `${pageBasePath}/${slug}.ipynb`;
      navigate('/' + encodePath(slugPath));
    }
  };

  const handleGoHome = () => {
    navigate('/');
  };

  // Extract notebook name from path (e.g., "Portfolio/Mutclust/1. Mutclust tutorial.ipynb" -> "mutclust")
  const getNotebookName = (path: string): string => {
    // Get the parent folder name (e.g., "Mutclust")
    const parts = path.split('/');
    if (parts.length >= 2) {
      return parts[parts.length - 2].toLowerCase();
    }
    const filename = parts.pop() || '';
    return filename.replace(/\.ipynb$/, '').toLowerCase();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-red-500">{error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <Sidebar
        structure={structure}
        selectedFile={selectedFile}
        onSelectFile={handleSelectFile}
        onGoHome={handleGoHome}
      />

      <main className="flex justify-center p-8">
        <div className="w-full max-w-3xl">
          {viewType === 'notebook' && notebook ? (
            <NotebookViewer
              notebook={notebook}
              notebookName={getNotebookName(selectedFile || '')}
            />
          ) : viewType === 'page' ? (
            <MarkdownViewer content={pageContent} onLinkClick={handlePageLinkClick} />
          ) : (
            <MarkdownViewer content={indexContent} />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
