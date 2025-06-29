import type { ReactNode } from 'react';

interface PageHeroProps {
  title: string;
  description: ReactNode;
}

const PageHero: React.FC<PageHeroProps> = ({ title, description }) => {
  return (
    <section className="mb-12 text-center animate-fadeIn">
      <h1 className="text-4xl md:text-5xl font-headline font-bold text-primary mb-4">{title}</h1>
      <div className="text-lg text-foreground/80 max-w-3xl mx-auto">
        {typeof description === 'string' ? <p>{description}</p> : description}
      </div>
    </section>
  );
};

export default PageHero;
