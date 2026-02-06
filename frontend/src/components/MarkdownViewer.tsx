import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ReactNode } from 'react';

interface MarkdownViewerProps {
  content: string;
  onLinkClick?: (href: string) => void;
}

// Generate slug from heading text
const generateSlug = (children: ReactNode): string => {
  const text = String(children).toLowerCase();
  return text
    .replace(/[^a-z0-9가-힣\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .trim();
};

export function MarkdownViewer({ content, onLinkClick }: MarkdownViewerProps) {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 id={generateSlug(children)}>{children}</h1>,
          h2: ({ children }) => <h2 id={generateSlug(children)}>{children}</h2>,
          h3: ({ children }) => <h3 id={generateSlug(children)}>{children}</h3>,
          h4: ({ children }) => <h4 id={generateSlug(children)}>{children}</h4>,
          a: ({ href, children }) => {
            if (onLinkClick && href) {
              return (
                <a
                  href={href}
                  onClick={(e) => {
                    e.preventDefault();
                    onLinkClick(href);
                  }}
                >
                  {children}
                </a>
              );
            }
            return <a href={href}>{children}</a>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
