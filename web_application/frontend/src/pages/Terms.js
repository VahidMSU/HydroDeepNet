import React from 'react';
import {
  TermsContainer,
  TermsTitle,
  ContentWrapper,
  SectionHeader,
  TermsText,
  ContactSection,
  Links,
} from '../styles/TermsConditions.tsx';

const Terms = () => {
  return (
    <TermsContainer>
      <TermsTitle>Terms and Conditions</TermsTitle>
      <ContentWrapper>
        <SectionHeader>1. Acceptance of Terms</SectionHeader>
        <TermsText>
          By accessing and using the HydroDeepNet platform, you acknowledge that you have read,
          understood, and agree to be bound by these Terms and Conditions. If you do not agree with
          any part of these terms, please do not use our services.
        </TermsText>

        <SectionHeader>2. Use of Services</SectionHeader>
        <TermsText>
          HydroDeepNet provides access to hydrological modeling tools, data analytics, and
          visualization services. You agree to use these services only for lawful purposes and in
          accordance with these Terms and Conditions.
        </TermsText>

        <SectionHeader>3. User Accounts</SectionHeader>
        <TermsText>
          When you create an account with us, you must provide accurate and complete information.
          You are responsible for safeguarding your password and for all activities that occur under
          your account.
        </TermsText>

        <SectionHeader>4. Data Usage and Privacy</SectionHeader>
        <TermsText>
          Our platform collects and processes hydrological data. By using our services, you grant us
          permission to use this data for improving our models and services. Please refer to our
          Privacy Policy for more details on how we handle your data.
        </TermsText>

        <SectionHeader>5. Intellectual Property</SectionHeader>
        <TermsText>
          All content, features, and functionality on HydroDeepNet, including but not limited to
          text, graphics, logos, icons, images, audio clips, digital downloads, data compilations,
          software, and the compilation thereof, are owned by HydroDeepNet or its licensors.
        </TermsText>

        <SectionHeader>6. Limitation of Liability</SectionHeader>
        <TermsText>
          HydroDeepNet shall not be liable for any indirect, incidental, special, consequential, or
          punitive damages, including loss of profits, data, or use, arising out of or in connection
          with these Terms or your use of the platform.
        </TermsText>

        <SectionHeader>7. Termination</SectionHeader>
        <TermsText>
          We may terminate or suspend your account and access to the platform immediately, without
          prior notice or liability, for any reason, including if you breach these Terms.
        </TermsText>

        <SectionHeader>8. Changes to Terms</SectionHeader>
        <TermsText>
          We reserve the right to modify these Terms at any time. Changes will be effective
          immediately upon posting on the platform. Your continued use of the platform after any
          modifications indicates your acceptance of the updated Terms.
        </TermsText>

        <SectionHeader>9. Governing Law</SectionHeader>
        <TermsText>
          These Terms shall be governed by and construed in accordance with the laws of the United
          States, without regard to its conflict of law provisions.
        </TermsText>

        <ContactSection>
          <TermsText>If you have any questions about these Terms, please contact us.</TermsText>
          <Links>
            <a href="/contact">Contact Us</a>
            <a href="/privacy">Privacy Policy</a>
          </Links>
        </ContactSection>
      </ContentWrapper>
    </TermsContainer>
  );
};

export default Terms;
