
import Image from 'next/image';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { 
  INTRO_HERO_TITLE, 
  INTRO_HERO_DESCRIPTION, 
  INTRO_HERO_CTA_TEXT, 
  INTRO_HERO_CTA_LINK,
  INTRO_FEATURES_SECTION_TITLE,
  INTRO_FEATURE_CARDS,
  INTRO_DATA_AI_SECTION_TITLE,
  INTRO_DATA_AI_CARD_DESCRIPTION,
  INTRO_DATA_AI_TEXT_P1,
  INTRO_DATA_AI_CHARACTERISTICS_LIST,
  INTRO_DATA_AI_MODELS_TEXT,
  INTRO_DATA_AI_ICON,
  INTRO_ACCURACY_SECTION_TITLE,
  INTRO_ACCURACY_CARD_DESCRIPTION,
  INTRO_ACCURACY_TEXT,
  INTRO_ACCURACY_CHART_TEXT,
  INTRO_ACCURACY_ICON,
  INTRO_TEAM_SECTION_TITLE,
  INTRO_TEAM_MEMBERS,
  INTRO_TEAM_ICON,
  PLACEHOLDER_HINTS
} from '@/lib/constants';

export default function IntroductionPage() {
  const DataAiIcon = INTRO_DATA_AI_ICON;
  const AccuracyIcon = INTRO_ACCURACY_ICON;
  const TeamIcon = INTRO_TEAM_ICON;

  return (
    <div className="space-y-12">
      <section className="bg-muted p-8 md:p-12 rounded-xl shadow-lg text-center animate-fadeIn">
        <h1 className="text-4xl md:text-5xl font-headline font-bold text-primary mb-4">{INTRO_HERO_TITLE}</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
          {INTRO_HERO_DESCRIPTION}
        </p>
        <Button asChild size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground">
          <Link href={INTRO_HERO_CTA_LINK}>{INTRO_HERO_CTA_TEXT}</Link>
        </Button>
      </section>

       <section className="bg-muted py-12 px-6 md:px-8 rounded-xl shadow-lg animate-fadeIn" style={{animationDelay: '0.2s'}}>
        <h2 className="text-3xl font-headline font-bold text-center text-primary mb-10">{INTRO_FEATURES_SECTION_TITLE}</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {INTRO_FEATURE_CARDS.map((feature, index) => {
            const Icon = feature.IconComponent;
            return (
              <Card key={index} className="shadow-lg hover:shadow-xl transition-shadow duration-300 animate-fadeIn bg-card" style={{animationDelay: `${0.3 + index * 0.1}s`}}>
                <CardHeader className="items-center text-center">
                  <Icon className="w-12 h-12 text-accent mb-3" />
                  <CardTitle className="font-headline text-xl text-primary">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-center text-card-foreground/80">
                  {feature.description}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      <section className="animate-fadeIn" style={{animationDelay: '0.6s'}}>
        <div className="grid md:grid-cols-2 gap-8 items-stretch">
          <Card className="shadow-lg flex flex-col bg-card">
            <CardHeader>
              <div className="flex items-center mb-2">
                <DataAiIcon className="w-8 h-8 text-primary mr-3" />
                <CardTitle className="font-headline text-2xl text-primary">{INTRO_DATA_AI_SECTION_TITLE}</CardTitle>
              </div>
              <CardDescription>{INTRO_DATA_AI_CARD_DESCRIPTION}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-card-foreground/90 flex-grow">
              <p dangerouslySetInnerHTML={{ __html: INTRO_DATA_AI_TEXT_P1 }} />
              <ul className="list-disc list-inside space-y-1 pl-4">
                {INTRO_DATA_AI_CHARACTERISTICS_LIST.map((item, index) => (
                  <li key={index} dangerouslySetInnerHTML={{ __html: item }} />
                ))}
              </ul>
              <p dangerouslySetInnerHTML={{ __html: INTRO_DATA_AI_MODELS_TEXT }} />
            </CardContent>
          </Card>
          <Card className="shadow-lg flex flex-col bg-card">
             <CardHeader>
              <div className="flex items-center mb-2">
                <AccuracyIcon className="w-8 h-8 text-primary mr-3" />
                <CardTitle className="font-headline text-2xl text-primary">{INTRO_ACCURACY_SECTION_TITLE}</CardTitle>
              </div>
              <CardDescription>{INTRO_ACCURACY_CARD_DESCRIPTION}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-card-foreground/90 flex-grow">
              <p dangerouslySetInnerHTML={{ __html: INTRO_ACCURACY_TEXT }} />
              <div className="bg-muted/50 p-4 rounded-md text-center border">
                <Image
                  src="/images/model_accuracy_chart.png"
                  alt="Model Accuracy Metrics Chart"
                  width={600}
                  height={300}
                  className="rounded-md mx-auto shadow-md object-contain"
                  data-ai-hint="metrics chart"
                />
                <p className="text-sm text-muted-foreground mt-2">{INTRO_ACCURACY_CHART_TEXT}</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      <section className="animate-fadeIn" style={{animationDelay: '0.8s'}}>
        <Card className="max-w-3xl mx-auto shadow-lg bg-card">
          <CardHeader>
            <div className="flex items-center justify-center mb-2">
              <TeamIcon className="w-8 h-8 text-primary mr-3" />
              <CardTitle className="font-headline text-2xl text-primary">{INTRO_TEAM_SECTION_TITLE}</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="text-card-foreground/90">
            <ul className="list-disc list-inside space-y-2 text-left sm:text-center">
              {INTRO_TEAM_MEMBERS.map((member, index) => (
                <li key={index}><strong>{member.name}:</strong> {member.role}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
